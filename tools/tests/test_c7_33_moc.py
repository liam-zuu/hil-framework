"""Tests of the 33-mark chain of c7_measure (check by rotation, mark-shift rule; design 2026-11-18_Thiet_ke_33_moc_
kiem_cheo_A3_v3.md section 5 (i)-(xii) and v4 section 6 (xiii)-(xvii)). SYNTHETIC detections projected from a known
camera here (NOT a measurement); the thresholds are test numbers (T 1.1 mm, T_raw 1.8 mm, Codex round 2 counterexample)."""
import copy
import importlib.util
import math
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
import c7_clip_probe as cp  # noqa: E402
import c7_measure as cm  # noqa: E402
import c7_moc_dich as md  # noqa: E402
from test_c7_clip_probe import W, H, truth_camera  # noqa: E402
from test_c7_measure import marks_dict, TAG_H, CAR_SIDE, OFFSET  # noqa: E402

CAM = truth_camera()
ZC = float(CAM.centre[2])
PLUMB = tuple(CAM.centre[:2])
MK = marks_dict(check=True)                     # ID 1-33 (25-33 taped for good: floor marks)
IDS = sorted(MK)
T, TR = 0.0011, 0.0018


def corners3(m, dxy=(0.0, 0.0)):
    c = cp.mark_corners_floor(m[0] + dxy[0], m[1] + dxy[1], m[2])
    return np.column_stack([c, np.zeros(4)])


def dets(cam=CAM, marks=MK, ids=IDS, noise=0.0, rng=None, moved=None, pix=None):
    """{id: (4, 2)} projected corners; moved {id: (dx, dy)} = the mark physically off its tape centre; pix {id: (du, dv)}
    = a pixel offset of the detection."""
    out = {}
    for i in ids:
        c = cam.project(corners3(marks[i], (moved or {}).get(i, (0.0, 0.0))))
        if noise:
            c = c + rng.normal(0.0, noise, c.shape)
        if pix and i in pix:
            c = c + np.asarray(pix[i])
        out[i] = c
    return out


def h_frames(n=10, noise=0.02, seed=1, **kw):
    rng = np.random.default_rng(seed)
    return [dets(noise=noise, rng=rng, **kw) for _ in range(n)]


def make_session(marks=MK, frames=None, model='B2'):
    frames = frames if frames is not None else h_frames(marks=marks)
    return cm.calibrate('H_syn.avi', marks, ZC, PLUMB, TAG_H, CAR_SIDE, floor=sorted(marks), model=model,
                        dets_in=(frames, (H, W), len(frames)))


@pytest.fixture(scope='module')
def sess():
    return make_session()


def fake_u_so(chains, t=T, tr=TR):
    ch = {c: dict(nguong_m=t, nguong_tho_m=tr, u_so_m=t - 0.001, u_so_tho_m=tr - 0.001, tap_tinh=[]) for c in chains}
    return dict(version='c7_u_so_v4', chuoi=ch, chi_tiet=dict(co_tap_luot=True))


def run_frames(n, fps=30.0, t0=0.0, noise=0.02, seed=5, moved_at=None, car=None):
    """[(idx, t, dets)]; moved_at(k) -> {id: (dx, dy)} per frame."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        d = dets(noise=noise, rng=rng, moved=moved_at(k) if moved_at else None)
        if car is not None:
            d[cm.CAR_ID] = car_px(*car)
        out.append((k, t0 + k / fps, d))
    return out


def car_px(x, y, psi, off=OFFSET):
    """Car tag corners in the image for the WHEEL pose (x, y, psi) and the tag offset (body frame)."""
    dx, dy, dpsi = off
    c, s = math.cos(psi), math.sin(psi)
    tx, ty = x + c * dx - s * dy, y + s * dx + c * dy
    body = cm.car_corners_body(CAR_SIDE)
    a = psi + dpsi
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    P = body @ R.T + np.array([tx, ty])
    return CAM.project(np.column_stack([P, np.full(4, TAG_H)]))


def series(fn, n, fps=60.0, t0=0.0):
    t = t0 + np.arange(n) / fps
    return t, np.array([fn(tt) for tt in t], float)


# ---------------------------------------------------------------- session block
def test_session_block(sess):
    kc = sess['kiem_cheo']
    assert sess['model_khoa'] == 'B2' and sess['chosen'] == 'B2'
    assert [len(g) for g in kc['nhom']] == [9, 8, 8, 8]
    assert sorted(i for g in kc['nhom'] for i in g) == IDS
    assert set(kc['cam_H']) == {'xe', '0', '1', '2', '3', '0,1', '0,2', '0,3', '1,2', '1,3', '2,3'}
    assert all(v['ok'] for v in kc['khop_H'].values())
    assert kc['nen_thieu'] == []
    assert set(kc['nen']) == {'xe', '0', '1', '2', '3'}
    assert len(kc['nen']['xe']) == 33 and all(len(kc['nen'][str(k)]) == 33 - len(kc['nhom'][k]) for k in range(4))
    assert kc['cond_H']['n'] == 10 and kc['cond_H']['max'] < cm.COND_MAX
    assert sess['vai_tro'] and set(sess['vai_tro']) == {str(i) for i in IDS}


def test_without_model_no_block():
    frames = h_frames()
    s = cm.calibrate('H_syn.avi', MK, ZC, PLUMB, TAG_H, CAR_SIDE, floor=IDS, dets_in=(frames, (H, W), len(frames)))
    assert 'kiem_cheo' not in s and 'model_khoa' not in s


# ---------------------------------------------------------------- (i) independence
def test_i_chain_k_does_not_depend_on_group_k():
    """Tape coordinates AND clip-H pixels of a mark of group 0 changed: every camera without group 0, every baseline of
    chain 0, and the chain-0 anchors, comparator deltas and check positions at fixed pixels are unchanged; the check
    error of that mark moves by exactly the tape change."""
    j = 1                                                          # group 0 = ids[0::4] = 1, 5, 9, ...
    mk2 = dict(MK)
    mk2[j] = (MK[j][0] + 0.005, MK[j][1] - 0.003, MK[j][2])
    fr = h_frames()
    fr2 = [dict(d) for d in fr]
    for d in fr2:
        d[j] = d[j] + np.array([2.0, -1.5])
    s1, s2 = make_session(MK, fr), make_session(mk2, fr2)
    k1, k2 = s1['kiem_cheo'], s2['kiem_cheo']
    assert k1['nhom'] == k2['nhom'] and j in k1['nhom'][0]
    for key in ('0', '0,1', '0,2', '0,3'):
        assert k1['cam_H'][key] == k2['cam_H'][key], key
    assert k1['cam_H']['xe'] != k2['cam_H']['xe']                  # the car camera does use the mark
    assert k1['nen']['0'] == k2['nen']['0']
    c1, c2 = cm.ChainCtx(s1), cm.ChainCtx(s2)
    d = dets(noise=0.02, rng=np.random.default_rng(3))
    f1, f2 = c1.frame(d), c2.frame(d)
    assert f1['neo']['0'] == f2['neo']['0']
    assert set(f1['delta']['0']) == set(f2['delta']['0'])
    for i in f1['delta']['0']:
        assert np.array_equal(f1['delta']['0'][i], f2['delta']['0'][i])
    r1 = {r['moc']: r for r in f1['kc'][0]['rows']}
    r2 = {r['moc']: r for r in f2['kc'][0]['rows']}
    assert r1[j]['x_doan'] == r2[j]['x_doan'] and r1[j]['y_doan'] == r2[j]['y_doan']
    assert r2[j]['ex'] == pytest.approx(r1[j]['ex'] - 0.005, abs=1e-12)
    assert r2[j]['ey'] == pytest.approx(r1[j]['ey'] + 0.003, abs=1e-12)


# ---------------------------------------------------------------- (ii) no chain reaction into chain k
def test_ii_extreme_mark_of_group_k_does_not_touch_E_k(sess):
    j = 6                                                          # group 1 (ids[1::4] = 2, 6, 10, ...)
    ctx = cm.ChainCtx(sess)
    assert ctx.gmap[j] == 1
    frames = run_frames(90, moved_at=lambda k: {j: (0.02, 0.0)} if k >= 30 else {})
    lich = cm.build_lich(sess, 'S', [('r1', 'luot', frames, 'D')], fake_u_so(ctx.chains), 'U')
    assert lich['chuoi']['1']['loai'] == {}                        # chain 1 never sees group 1
    for c in ('xe', '0', '2', '3'):
        assert list(lich['chuoi'][c]['loai']) == [str(j)], c
        assert lich['chuoi'][c]['loai'][str(j)]['t_s'] <= 1.0 + 1e-9
        assert lich['chuoi'][c]['nen_doi'] == []
        assert not [m for m in lich['chuoi'][c]['moc_nghi'] if m[0] == j]     # excluded: never a moc_nghi after
    # the 20 mm anchor moved another mark's comparator of chain 0 by about 1.7 mm (two alarms): explained by mark 6
    assert [g for g in lich['giai_thich_nen_doi'] if g['chap_nhan']] == \
        [dict(chuoi='0', moc=j, tu=g['tu'], dinh_m=g['dinh_m'], chap_nhan=True) for g in lich['giai_thich_nen_doi']
         if g['chap_nhan']]
    assert any(g['chuoi'] == '0' and g['moc'] == j and g['chap_nhan'] for g in lich['giai_thich_nen_doi'])


# ---------------------------------------------------------------- pure rule: (iii) (iv) (vi) (xiii) (xiv) (xv)
def clip_of(t, dj, name='c', kind='luot'):
    return md.Clip(name, kind, t, dj)


def test_iii_step_of_1_5_mm_is_alarmed_and_t_bd_is_before_it():
    t, d = series(lambda tt: [0.0015 if tt >= 1.0 else 0.0, 0.0], 180)
    s = md.schedule([clip_of(t, {7: d})], T, TR, [7])
    assert list(s['loai']) == ['7']
    v = s['loai']['7']
    assert v['t_s'] <= 1.0 and not v['bat_dinh']
    assert t[v['bao_dong'][1]] > 1.0
    assert s['moc_nghi'] == [] and s['nen_doi'] == []


def test_iv_short_shift_is_not_alarmed_and_is_within_the_raw_bound():
    t, d = series(lambda tt: [0.0015 if 1.0 <= tt < 1.1 else 0.0, 0.0], 180)
    s = md.schedule([clip_of(t, {7: d})], T, TR, [7])
    assert s['loai'] == {} and s['moc_nghi'] == []
    u = fake_u_so(['xe'])
    assert np.max(np.linalg.norm(d, axis=1)) <= cm.undetected_bound(u)


def test_raw_spike_is_a_moc_nghi_only():
    t, d = series(lambda tt: [0.0025 if abs(tt - 1.0) < 1e-6 else 0.0, 0.0], 120)
    s = md.schedule([clip_of(t, {7: d})], T, TR, [7])
    assert s['loai'] == {} and s['moc_nghi'] == [[7, 0, 60]]


def test_vi_two_marks_at_once_is_a_base_change_not_an_exclusion():
    f = lambda tt: [0.0015 if 1.0 <= tt < 2.0 else 0.0, 0.0]       # noqa: E731
    t, d = series(f, 300)
    z = np.zeros_like(d)
    s = md.schedule([clip_of(t, {3: d, 4: d, 5: z})], T, TR, [3, 4, 5])
    assert s['loai'] == {}
    assert len(s['nen_doi']) == 1
    e = s['nen_doi'][0]
    assert set(e['moc']) == {3, 4}
    assert t[e['tu'][1]] <= 1.0                                    # from the last base check
    assert e['den'] is not None and t[e['den'][1]] >= 2.0          # closed by a 1 s span with both back to <= T
    assert md.inconclusive_at(s, 0, 120)[0] and not md.inconclusive_at(s, 0, 290)[0]


def test_vi_two_alarms_half_a_second_apart_are_a_base_change():
    t, d3 = series(lambda tt: [0.0015 if 1.0 <= tt < 2.5 else 0.0, 0.0], 300)
    _, d4 = series(lambda tt: [0.0015 if 1.5 <= tt < 2.5 else 0.0, 0.0], 300)
    s = md.schedule([clip_of(t, {3: d3, 4: d4})], T, TR, [3, 4])
    assert s['loai'] == {} and len(s['nen_doi']) == 1 and set(s['nen_doi'][0]['moc']) == {3, 4}
    _, d4 = series(lambda tt: [0.0015 if 2.6 <= tt < 3.5 else 0.0, 0.0], 300)          # 1.6 s later: two exclusions
    s = md.schedule([clip_of(t, {3: d3, 4: d4})], T, TR, [3, 4])
    assert sorted(s['loai']) == ['3', '4'] and s['nen_doi'] == []


def test_mark_centre_is_the_mean_of_the_back_projected_corners():
    cam = truth_camera('B2', k=[0.25, 0.1], tilt_deg=(12.0, -9.0, 4.0))
    c = cam.project(corners3((1.3, 0.85, 0.15)))
    got = cm.mark_centre_floor(cam, c)
    assert np.allclose(got, cam.pix2floor(c).mean(axis=0), atol=1e-12)
    assert np.allclose(got, [1.3, 0.85], atol=1e-9)
    old = cam.pix2floor(c.mean(axis=0, keepdims=True))[0]          # d59197a: back-projected mean of the image corners
    assert np.linalg.norm(old - got) > 1e-5


def test_xiii_four_directions_are_alarmed_by_the_median_magnitude():
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    t, d = series(lambda tt: [0.0, 0.0], 180)
    for e in range(60, 180):
        d[e] = 0.0015 * np.array(dirs[e % 4])
    assert np.linalg.norm(np.median(d[100:130], axis=0)) < 1e-12     # the vector median sees nothing
    s = md.schedule([clip_of(t, {7: d})], T, TR, [7])
    assert list(s['loai']) == ['7']


def test_xiv_shift_at_the_end_of_a_clip_alarmed_in_the_next_one():
    t1, d1 = series(lambda tt: [0.0015 if tt >= 1.9 else 0.0, 0.0], 120)        # 0.1 s at the end of clip 1
    t2, d2 = series(lambda tt: [0.0015, 0.0], 120, t0=10.0)
    s = md.schedule([clip_of(t1, {7: d1}, 'a'), clip_of(t2, {7: d2}, 'b')], T, TR, [7])
    v = s['loai']['7']
    assert v['tu'][0] == 0 and t1[v['tu'][1]] <= 1.9                # the exclusion covers the shifted tail
    assert v['bat_dinh'] and v['bao_dong'][0] == 1
    assert s['bat_dinh_moc'] == [dict(moc=7, tu=v['tu'], den=v['bao_dong'])]
    assert md.inconclusive_at(s, 0, 119)[1] == [7] and md.inconclusive_at(s, 1, 5)[1] == [7]
    assert md.inconclusive_at(s, 0, 0)[1] == []


def test_v_shift_at_a_clip_start_never_stable_before():
    t2, d2 = series(lambda tt: [0.0015, 0.0], 120)
    s = md.schedule([clip_of(t2, {7: d2})], T, TR, [7])
    v = s['loai']['7']
    assert v['tu'] == [0, 0] and v['bat_dinh']


def test_xv_base_change_stays_open_while_an_alarmed_mark_is_hidden():
    t, d = series(lambda tt: [0.0015 if 1.0 <= tt < 2.0 else 0.0, 0.0], 300)
    hid = d.copy()
    hid[t >= 2.0] = np.nan                                         # mark 4 hidden after the base change
    s = md.schedule([clip_of(t, {3: d, 4: hid})], T, TR, [3, 4])
    assert len(s['nen_doi']) == 1 and s['nen_doi'][0]['den'] is None
    assert md.inconclusive_at(s, 0, 299)[0]


def test_clip_h_closes_a_base_change():
    t, d = series(lambda tt: [0.0015 if tt >= 1.0 else 0.0, 0.0], 150)          # never back inside the run clip
    th, dh = series(lambda tt: [0.0, 0.0], 120, t0=20.0)
    s = md.schedule([clip_of(t, {3: d, 4: d}, 'r'), clip_of(th, {3: dh, 4: dh}, 'H_m', 'H')], T, TR, [3, 4])
    assert s['nen_doi'][0]['den'] == [1, 0]


def test_window_restarts_after_a_gap():
    t = np.arange(40) / 60.0
    r = np.zeros(40)
    r[15:22] = np.nan                                              # 0.117 s without sample
    m, ws, sg = md.windows(t, r)
    assert sg[14] == 0 and sg[22] == 1
    assert np.isnan(m[22]) and np.isfinite(m[31]) and ws[31] == 22


# ---------------------------------------------------------------- (vii) (xvi) rescoring and convergence
@pytest.fixture(scope='module')
def spike_run(sess):
    j = 11
    frames = run_frames(60, moved_at=lambda k: {j: (0.003, 0.0)} if k == 40 else {},
                        car=(0.2, -0.1, math.radians(20.0)))
    return j, frames


def test_vii_xvi_moc_nghi_changes_the_schedule_and_the_track(sess, spike_run):
    j, frames = spike_run
    ctx = cm.ChainCtx(sess)
    u = fake_u_so(ctx.chains)
    runs = [('r1', 'luot', frames, 'D')]
    L0 = cm.build_lich(sess, 'S', runs)                            # step 1: no thresholds yet
    assert all(v == dict(loai={}, moc_nghi=[], nen_doi=[], bat_dinh_moc=[]) for v in L0['chuoi'].values())
    L1 = cm.build_lich(sess, 'S', runs, u, 'U', L0)
    assert not L1['hoi_tu']
    assert [j, 0, 40] in L1['chuoi']['xe']['moc_nghi'] and L1['chuoi']['xe']['loai'] == {}
    L2 = cm.build_lich(sess, 'S', runs, u, 'U', L1)
    assert L2['hoi_tu'] and L2['lich_sha256'] == L1['lich_sha256'] and len(L2['lap']) == 3
    r0, _, i0 = cm.track_chains(frames, 'r1.mov', 'V', sess, OFFSET, run='r1')
    r2, _, i2 = cm.track_chains(frames, 'r1.mov', 'V', sess, OFFSET, lich=L2, run='r1', u_so=u, u_so_sha='U')
    assert i0['cham_lai'] == 1 and i2['cham_lai'] == 2 and i2['lich_sha256'] == L2['lich_sha256']
    assert str(j) not in r2[40]['moc_dung'].split(';') and str(j) in r0[40]['moc_dung'].split(';')
    for k in range(40):                                            # before the first change: the same poses
        assert r2[k]['x'] == r0[k]['x'] and r2[k]['rms_px'] == r0[k]['rms_px']
    assert {'chuoi': 'xe', 'id': j, 'khung': 40} in i2['moc_nghi']
    assert i2['nguong_dung'] == {c: dict(nguong_m=T, nguong_tho_m=TR) for c in ctx.chains}


def test_viii_a_cycle_is_not_converged(sess, spike_run):
    _, frames = spike_run
    ctx = cm.ChainCtx(sess)
    u = fake_u_so(ctx.chains)
    runs = [('r1', 'luot', frames, 'D')]
    L0 = cm.build_lich(sess, 'S', runs)
    L1 = cm.build_lich(sess, 'S', runs, u, 'U', L0)
    fake = copy.deepcopy(L0)                                       # same exclusions as L0, history [L1, Z]
    fake.update(tap_moc=L1['tap_moc'], lich_sha256='Z', lap=[dict(lich_sha256=L1['lich_sha256']), dict(lich_sha256='Z')])
    L = cm.build_lich(sess, 'S', runs, u, 'U', fake)
    assert L['lich_sha256'] == L1['lich_sha256'] and L['chu_trinh'] and not L['hoi_tu']
    assert 'chua ket luan' in L['ket_luan']


def test_lich_refuses_u_so_without_the_previous_sets(sess, spike_run):
    _, frames = spike_run
    ctx = cm.ChainCtx(sess)
    runs = [('r1', 'luot', frames, 'D')]
    L0 = cm.build_lich(sess, 'S', runs)
    u = fake_u_so(ctx.chains)
    u['chi_tiet']['co_tap_luot'] = False
    with pytest.raises(ValueError, match='tap-moc'):
        cm.build_lich(sess, 'S', runs, u, 'U', L0)
    with pytest.raises(ValueError, match='other session'):
        cm.build_lich(sess, 'S2', runs, fake_u_so(ctx.chains), 'U', L0)


# ---------------------------------------------------------------- (ix) fit flags
def test_ix_fit_flags(sess):
    ctx = cm.ChainCtx(sess)
    d = dets()
    cam0 = ctx.cams['xe']
    line = [3, 4, 5]                                               # three marks on one row + one: poor geometry
    obj = np.vstack([cp.mark_corners_floor(*MK[i]) for i in line])
    _, _, ok, ly, cond = cm.refit_pose_checked(cam0, obj, np.vstack([d[i] for i in line]), ctx.zc_h, cond_max=10.0)
    assert not ok and ly == 'dieu_kien' and cond > 10.0
    obj = np.vstack([cp.mark_corners_floor(*MK[i]) for i in IDS])
    _, _, ok, ly, _ = cm.refit_pose_checked(cam0, obj, np.vstack([d[i] for i in IDS]), ctx.zc_h + 0.5)
    assert not ok and ly == 'khop_khong_tin'
    few = {i: d[i] for i in (1, 2, 3)}
    f = ctx.frame(few)
    assert not f['main'][2] and f['main'][3] == 'thieu_moc' and f['delta']['xe'] == {}
    assert f['kc'][0]['ly_do'] == 'thieu_moc' and not f['kc'][0]['ok']
    rows, _, info = cm.track_chains([(0, 0.0, few)], 'x', 'V', sess)
    assert 'khop' in rows[0]['why'] and info['n_khop_khong_tin'] == 1


# ---------------------------------------------------------------- (x) baseline against itself
def test_x_baseline_compared_with_its_own_clip(sess):
    ctx = cm.ChainCtx(sess)
    fr = h_frames()                                                # the very clip-H frames of the session
    dl = {}
    for d in fr:
        for c, v in ctx.frame(d)['delta'].items():
            for j, x in v.items():
                dl.setdefault((c, j), []).append(x)
    means = [np.linalg.norm(np.mean(v, axis=0)) for v in dl.values()]
    maxs = [max(np.linalg.norm(v, axis=1)) for v in dl.values()]
    assert max(means) < 1e-9                                       # the baseline is the mean of these positions
    assert min(maxs) > 0.0 and max(maxs) < T                       # each frame deviates, not 0


# ---------------------------------------------------------------- (xi) (xvii) sensitivity
@pytest.mark.parametrize('off', [(0.0, 0.0, 0.0), OFFSET])
def test_xi_xvii_sensitivity_matches_a_refit_and_uses_the_wheel_centre(sess, off):
    ctx = cm.ChainCtx(sess)
    d = dets()
    pose = (0.3, -0.2, math.radians(35.0))
    cpx = car_px(*pose, off=off)
    obj = np.vstack([cp.mark_corners_floor(*MK[i]) for i in IDS])
    cam, *_ = cm.refit_pose_checked(ctx.cams['xe'], obj, np.vstack([d[i] for i in IDS]), ctx.zc_h)
    D = 0.002
    s = cm.car_sensitivity(cam, IDS, d, MK, cpx, CAR_SIDE, sess['kappa'], PLUMB, off, D, vertices=True)
    assert 0.8 < s['ratio_xy'] < 1.25 and 0.8 < s['ratio_psi'] < 1.25
    assert 0.0 < s['U_xy'] < 33 * D and s['U_psi'] > 0.0
    p0 = cm.car_pose(cam, cpx, CAR_SIDE, sess['kappa'], PLUMB, off)
    assert math.hypot(p0['x'] - pose[0], p0['y'] - pose[1]) < 1e-5          # the wheel centre, not the tag centre
    if off != (0.0, 0.0, 0.0):
        s0 = cm.car_sensitivity(cam, IDS, d, MK, car_px(*pose, off=(0.0, 0.0, 0.0)), CAR_SIDE, sess['kappa'], PLUMB,
                                (0.0, 0.0, 0.0), D)
        assert s['U_xy'] != pytest.approx(s0['U_xy'], rel=1e-6)


def test_final_track_writes_the_sensitivity(sess, spike_run):
    j, frames = spike_run
    ctx = cm.ChainCtx(sess)
    u = fake_u_so(ctx.chains)
    runs = [('r1', 'luot', frames, 'D')]
    L1 = cm.build_lich(sess, 'S', runs, u, 'U', cm.build_lich(sess, 'S', runs))
    rows, khung, info = cm.track_chains(frames, 'r1.mov', 'V', sess, OFFSET, lich=L1, run='r1', u_so=u, u_so_sha='U')
    v = [r for r in rows if r['valid']]
    assert v and all(r['U_moc_xy_m'] > 0 and r['U_moc_psi_rad'] > 0 for r in v)
    assert info['do_nhay']['D_m'] == pytest.approx(cm.undetected_bound(u))
    assert {r['chuoi'] for r in khung} == {'0', '1', '2', '3'}
    assert all(ctx.gmap[r['moc']] == r['nhom'] == int(r['chuoi']) for r in khung)


# ---------------------------------------------------------------- CLI round trip, file contract with KIEMCHEO
def test_cli_track_lich_track(tmp_path, sess, spike_run, monkeypatch):
    import json
    _, frames = spike_run
    sp = tmp_path / 'session.json'
    sp.write_text(json.dumps(cm._json(sess)), encoding='utf-8')
    d1 = tmp_path / 'r1'
    d1.mkdir()
    cm.save_dets(str(d1 / 'dets.npz'), frames, 'V', (H, W))
    assert cm.main(['track', '--dets', str(d1 / 'dets.npz'), '--session', str(sp), '--out', str(tmp_path / 't1')]) == 0
    tr = json.loads((tmp_path / 't1' / 'track.json').read_text(encoding='utf-8'))
    assert tr['cham_lai'] == 1 and tr['model_khoa'] == 'B2' and tr['nhom'] == sess['kiem_cheo']['nhom']
    fr = cm.read_frames(str(tmp_path / 't1' / 'frames.csv'))
    need = {'moc_thay', 'moc_dung'} | {f'moc_neo_{c}' for c in ('xe', '0', '1', '2', '3')} | \
        {f'kc_{n}_{k}' for n in ('ok', 'n', 'err_max', 'ly_do') for k in range(4)}
    assert need <= set(fr[0])
    assert cm.main(['lich', '--session', str(sp), '--luot', f'r1={d1}', '--out', str(tmp_path / 'L0')]) == 0
    u = fake_u_so(['xe', '0', '1', '2', '3'])
    (tmp_path / 'u.json').write_text(json.dumps(u), encoding='utf-8')
    for a, b in (('L0', 'L1'), ('L1', 'L2')):
        assert cm.main(['lich', '--session', str(sp), '--luot', f'r1={d1}', '--u-so', str(tmp_path / 'u.json'),
                        '--truoc', str(tmp_path / a / 'lich.json'), '--out', str(tmp_path / b)]) == 0
    L2 = json.loads((tmp_path / 'L2' / 'lich.json').read_text(encoding='utf-8'))
    assert L2['hoi_tu']
    tap = json.loads((tmp_path / 'L2' / 'tap_moc.json').read_text(encoding='utf-8'))
    assert set(tap) == {'xe', '0', '1', '2', '3'} and all(isinstance(S, list) for v in tap.values() for S in v)
    # the final pass refuses a schedule not converged, another u_so, or other detections
    args = ['track', '--dets', str(d1 / 'dets.npz'), '--session', str(sp), '--run', 'r1', '--u-so', str(tmp_path / 'u.json')]
    assert cm.main(args + ['--lich', str(tmp_path / 'L1' / 'lich.json'), '--out', str(tmp_path / 'x1')]) == 2
    assert cm.main(args + ['--lich', str(tmp_path / 'L2' / 'lich.json'), '--out', str(tmp_path / 't2')]) == 0
    tr2 = json.loads((tmp_path / 't2' / 'track.json').read_text(encoding='utf-8'))
    assert tr2['cham_lai'] == 2 and tr2['u_so_sha256'] == cm.sha256(str(tmp_path / 'u.json'))
    assert tr2['frames_sha256'] == cm.sha256(str(tmp_path / 't2' / 'frames.csv'))
    assert tr2['kiem_cheo_khung_sha256'] == cm.sha256(str(tmp_path / 't2' / 'kiem_cheo_khung.csv'))
    # regate keeps the 33-mark columns and the khop gate
    assert cm.main(['regate', '--frames', str(tmp_path / 't2' / 'frames.csv'), '--session', str(sp),
                    '--out', str(tmp_path / 'rg')]) == 0
    rg = cm.read_frames(str(tmp_path / 'rg' / 'frames.csv'))
    assert [r['moc_dung'] for r in rg] == [r['moc_dung'] for r in cm.read_frames(str(tmp_path / 't2' / 'frames.csv'))]


def test_regate_keeps_the_khop_gate(tmp_path, sess):
    rows, _, _ = cm.track_chains([(0, 0.0, {i: dets()[i] for i in (1, 2, 3)})], 'x', 'V', sess)
    cm.write_frames(str(tmp_path / 'f.csv'), rows)
    back = cm.read_frames(str(tmp_path / 'f.csv'))
    assert back[0]['khop_ok'] is False
    cm.regate(back, sess)
    assert 'khop' in back[0]['why']


# ---------------------------------------------------------------- (xii) an old session tracks as d59197a
def _old_module(tmp_path):
    root = subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True, cwd=HERE).stdout.strip()
    src = subprocess.run(['git', 'show', 'd59197a:tools/c7_measure.py'], capture_output=True, cwd=root)
    if src.returncode:
        pytest.skip('commit d59197a not in this repo')
    p = tmp_path / 'c7_measure_d59197a.py'
    p.write_bytes(src.stdout)
    spec = importlib.util.spec_from_file_location('c7_measure_d59197a', str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_xii_old_session_tracks_as_d59197a(tmp_path):
    from test_c7_measure import render_floor, floor_texture, car_texture, render_car, write_video, poses_line
    old = _old_module(tmp_path)
    tex = floor_texture(check=False)
    base = render_floor(CAM, tex)
    write_video(tmp_path / 'H.avi', [render_floor(CAM, floor_texture(check=True))] * 4)
    car = car_texture()
    frames = []
    for p in poses_line(8):
        f = base.copy()
        render_car(f, CAM, car, p)
        frames.append(f)
    write_video(tmp_path / 'run.avi', frames)
    s = cm.calibrate(str(tmp_path / 'H.avi'), marks_dict(check=True), ZC, PLUMB, TAG_H, CAR_SIDE)
    assert 'kiem_cheo' not in s
    new_rows, _ = cm.track(str(tmp_path / 'run.avi'), s, tag_offset=OFFSET)
    old_rows, _ = old.track(str(tmp_path / 'run.avi'), s, tag_offset=OFFSET)
    keys = ('x', 'y', 'psi', 'rms_px', 'n_marks', 'valid', 'res_m', 'scale')
    for a, b in zip(new_rows, old_rows):
        for k in keys:
            assert a[k] == b[k] or (a[k] != a[k] and b[k] != b[k]), k


# ---------------------------------------------------------------- c7_usb_fetch.check_marks
def test_check_marks_uses_the_session_floor_ids():
    import c7_usb_fetch as uf
    seen = {i: (100.0 + i, 100.0) for i in range(1, 34)}
    s = dict(height=1080, width=1920, ref_centres_px={str(i): [100.0 + i, 100.0] for i in range(1, 34)},
             floor_ids=list(range(1, 34)))
    out = uf.check_marks(seen, (1080, 1920), s)
    assert out['n_floor'] == 33 and out['n_check'] == 0 and out['verdict'] == 'OK'
    out = uf.check_marks(seen, (1080, 1920), None)
    assert out['n_floor'] == 24 and out['n_check'] == 9
