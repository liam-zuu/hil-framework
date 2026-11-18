"""Tests for tools/c7_nhac_tam.py (plate lift test of C8, design 2026-11-18_Thiet_ke_c7_nhac_tam_v4.md section 4).
Synthetic data and clips rendered here from a known camera (NOT a measurement). The floor scene reuses the mark
layout and the camera of test_c7_clip_probe; the car (body with textured tape, plate with the tag) is drawn here."""
import json
import math
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import c7_clip_probe as cp  # noqa: E402
import c7_measure as cm  # noqa: E402
import c7_nhac_tam as nt  # noqa: E402
from test_c7_clip_probe import MARKS, SIDES, W, H, FPS, truth_camera  # noqa: E402

TAG_H = 0.12
CAR_SIDE = 0.15
BODY_L, BODY_W = 0.40, 0.34                  # synthetic body, larger than the car so the boxes sit well inside
PLATE = 0.18                                 # tag 0.15 + white margin 0.015 each side
BOX_BODY = [(0.15, 0.125), (-0.15, -0.125)]  # tape centres in the body frame (outside the plate)
CAM = truth_camera()
PLUMB = tuple(CAM.centre[:2])
ZC = float(CAM.centre[2])
TEST_CFG = dict(trim_s=0.0, min_frames_s1=3, min_frames_abs=3, r_min_frames_s1=3, r_min_frames_abs=3, min_frac=0.5)
CFG = dict(nt.DEFAULTS)
R_OK = dict(dat=True)


# ---------------------------------------------------------------- unit: analysis on synthetic summaries
def rot(a):
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def summaries(body, plate, boxes_on_plate=False, bend=None, cam=(0.0, 0.0, 2.9)):
    """body[j] = (x, y, psi) body pose, plate[j] = (dx, dy, dpsi) tag centre / heading in the body frame."""
    out = []
    bb = [np.array([0.13, 0.11]), np.array([-0.13, -0.11]), np.array([0.13, -0.11])]
    for j, ((x, y, a), (dx, dy, dp)) in enumerate(zip(body, plate)):
        R, t = rot(a), np.array([x, y])
        tag = R @ np.array([dx, dy]) + t
        if boxes_on_plate:            # tape stuck on the plate: it moves with the tag
            pts = [rot(a + dp) @ b + tag for b in bb]
        else:
            pts = [R @ b + t for b in bb]
        if bend is not None and j == bend[0]:
            pts[0] = pts[0] + np.array([bend[1], 0.0])
        out.append(dict(ok=True, tag=[tag[0], tag[1], nt.wrap(a + dp)], box=[p.tolist() for p in pts], cam=list(cam),
                        scale=1.0))
    return out


PLATE_OFF = [(0.0, 0.0, 0.0), (0.0008, 0.0, math.radians(0.3)), (-0.0003, 0.0005, math.radians(-0.1)),
             (0.0, -0.0004, math.radians(0.15)), (0.0002, 0.0002, math.radians(0.05))]
STILL = [(0.3, -0.2, math.radians(40.0))] * 5


def truth_rep(plate, eps=None):
    eps = eps or [0.0] * len(plate)
    n = range(len(plate))
    best_p = max(math.hypot(plate[j][0] - plate[k][0], plate[j][1] - plate[k][1]) + eps[j] + eps[k]
                 for j in n for k in n if j != k)
    best_a = max(abs(math.degrees(plate[j][2] - plate[k][2])) for j in n for k in n)
    return best_p, best_a


def test_body_pushed_is_removed_with_eps():
    body = [(0.0, 0.0, 0.0), (0.002, -0.001, math.radians(0.5)), (0.0015, 0.0, math.radians(-0.4)),
            (0.0, 0.002, math.radians(0.2)), (0.001, 0.001, 0.0)]
    res = nt.analyse(summaries(body, PLATE_OFF), CFG, R_OK, tag_h=0.12)
    eps = [math.hypot(b[0], b[1]) * CFG['dh_m'] / (2.9 - 0.12) for b in body]
    tp, ta = truth_rep(PLATE_OFF, eps)
    assert res['kiem_than'] and res['so_dung'] == 'tru_than'
    assert res['tru_than']['vi_tri_m'] == pytest.approx(tp, abs=1e-9)
    assert res['tru_than']['goc_deg'] == pytest.approx(ta, abs=1e-9)
    assert res['vi_tri_m'] == res['tru_than']['vi_tri_m'] and res['goc_deg'] == res['tru_than']['goc_deg']
    assert res['tho']['vi_tri_m'] > tp
    assert res['than_dich_max_m'] == pytest.approx(max(math.hypot(b[0], b[1]) for b in body), abs=1e-9)
    assert res['than_dich_max_deg'] == pytest.approx(0.5, abs=1e-9)


def test_still_body_gives_the_plate_repeatability():
    res = nt.analyse(summaries(STILL, PLATE_OFF), CFG, R_OK)
    tp, ta = truth_rep(PLATE_OFF)
    assert res['kiem_than']
    assert res['vi_tri_m'] == pytest.approx(tp, abs=1e-9) and res['goc_deg'] == pytest.approx(ta, abs=1e-9)
    assert res['than_dich_max_m'] == pytest.approx(0.0, abs=1e-12)


def test_nt3_inverse_of_rigid_fit():
    """Codex NT-3: body turned 0.2 deg about c = (0.1, 0) (t = (R - I) c), tag still on the floor: the tag moved
    0.349 mm relative to the body; raw says 0."""
    a = math.radians(0.2)
    c = np.array([0.1, 0.0])
    t = (rot(a) - np.eye(2)) @ c
    bb = [np.array([0.13, 0.11]), np.array([-0.13, -0.11])]
    s0 = dict(tag=[0.0, 0.0, 0.0], box=[b.tolist() for b in bb], cam=[0, 0, 2.9], scale=1.0)
    s1 = dict(tag=[0.0, 0.0, 0.0], box=[(rot(a) @ b + t).tolist() for b in bb], cam=[0, 0, 2.9], scale=1.0)
    res = nt.analyse([s0, s1, s0, s0], dict(CFG, dh_m=0.0), R_OK)
    assert res['tho']['vi_tri_m'] == pytest.approx(0.0, abs=1e-12)
    assert res['tru_than']['vi_tri_m'] == pytest.approx(0.000349, abs=2e-6)
    assert res['vi_tri_m'] == res['tru_than']['vi_tri_m']


def test_opposite_body_and_plate_motion():
    """Body -0.8 mm, plate +0.8 mm relative to the body: the tag stays put on the floor (raw 0); body removed finds it."""
    body = [(0.0, 0.0, 0.0), (-0.0008, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    plate = [(0.0, 0.0, 0.0), (0.0008, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    res = nt.analyse(summaries(body, plate), dict(CFG, dh_m=0.0), R_OK)
    assert res['tho']['vi_tri_m'] == pytest.approx(0.0, abs=1e-12)
    assert res['vi_tri_m'] == pytest.approx(0.0008, abs=1e-9)


def test_tape_on_the_plate_without_clip_r_proof_falls_back_to_raw():
    """Tape on the plate: body removed ~0. Clip R fails for such boxes (unit below) -> kiem_than false, raw value."""
    body = [(0.0, 0.0, 0.0)] * 5
    res = nt.analyse(summaries(body, PLATE_OFF, boxes_on_plate=True), CFG, dict(dat=False))
    tp, ta = truth_rep(PLATE_OFF)
    assert res['tru_than']['vi_tri_m'] < 3e-5                         # only eps (about 1 % of the plate motion)
    assert not res['kiem_than'] and 'clip_R_khong_dat' in res['ly_do_khong_kiem_than'] and res['so_dung'] == 'tho'
    assert res['vi_tri_m'] == pytest.approx(tp, abs=1e-9) and res['goc_deg'] == pytest.approx(ta, abs=1e-9)
    res = nt.analyse(summaries(body, PLATE_OFF), CFG, None)
    assert not res['kiem_than'] and res['ly_do_khong_kiem_than'] == ['khong_co_clip_R']


def test_non_rigid_box_turns_the_body_check_off():
    res = nt.analyse(summaries(STILL, PLATE_OFF, bend=(2, 0.0015)), CFG, R_OK)
    assert not res['kiem_than'] and res['so_dung'] == 'tho'
    assert any(w.startswith('N2_du_khop') for w in res['ly_do_khong_kiem_than'])


def test_one_box_is_not_a_body_check():
    s = summaries(STILL, PLATE_OFF)
    for x in s:
        x['box'] = x['box'][:1]
    res = nt.analyse(s, CFG, R_OK)
    assert not res['kiem_than'] and res['ly_do_khong_kiem_than'] == ['it_hon_2_vung'] and res['tru_than'] is None


def test_camera_shift_enters_eps():
    s = summaries(STILL, PLATE_OFF)
    s[3]['cam'] = [0.005, 0.0, 2.9]
    res = nt.analyse(s, CFG, R_OK, tag_h=0.12)
    eps = [0.0, 0.0, 0.0, 0.005 * CFG['dh_m'] / (2.9 - 0.12), 0.0]                 # dH = 0
    assert res['than'][3]['camera_dich_xy_m'] == pytest.approx(0.005)
    assert res['vi_tri_m'] == pytest.approx(truth_rep(PLATE_OFF, eps)[0], abs=1e-9)


def test_angles_wrap_around_180_raw_and_body_removed():
    poses = [[0, 0, math.radians(179.9)], [0, 0, math.radians(-179.9)], [0, 0, math.pi], [0, 0, -math.pi]]
    assert nt.pairwise(poses)['goc_deg'] == pytest.approx(0.2, abs=1e-9)
    assert math.degrees(nt.circ_median([math.radians(179.0), math.radians(-179.0), math.radians(179.5)])) \
        == pytest.approx(179.5, abs=1e-9)
    body = [(0.0, 0.0, math.radians(179.95)), (0.0, 0.0, math.radians(-179.95)), (0.0, 0.0, math.pi),
            (0.0, 0.0, math.radians(179.9))]
    plate = [(0.0, 0.0, 0.0), (0.0, 0.0, math.radians(0.1)), (0.0, 0.0, 0.0), (0.0, 0.0, math.radians(-0.05))]
    res = nt.analyse(summaries(body, plate), CFG, R_OK)
    assert res['tru_than']['goc_deg'] == pytest.approx(0.15, abs=1e-6)
    assert res['tho']['goc_deg'] < 1.0


def test_nt2_1_body_turn_seen_through_a_height_error():
    """Codex NT2-1: boxes really at z_b = 0.09 m, projected at h = 0.12 m, camera C off O; body turned 1 deg with the
    translation that makes the projected box motion a pure rotation (t = 0); tag still at O turning with the body.
    The tag moved relative to the body; eps must cover it."""
    H, h, zb = 2.632, 0.12, 0.09
    C = np.array([-0.4802, -0.0329])
    a = (H - h) / (H - zb)
    R = rot(math.radians(1.0))
    T = -(1 - a) * (np.eye(2) - R) @ C / a
    q = [np.array([0.13, 0.11]), np.array([-0.13, -0.11])]
    proj = lambda P: C + a * (P - C)                                    # noqa: E731
    box0 = [proj(x).tolist() for x in q]
    box1 = [proj(R @ x + T).tolist() for x in q]
    cam = [C[0], C[1], H]
    s0 = dict(tag=[0.0, 0.0, 0.0], box=box0, cam=cam, scale=1.0)
    s1 = dict(tag=[0.0, 0.0, math.radians(1.0)], box=box1, cam=cam, scale=1.0)
    truth = float(np.hypot(*(R.T @ (np.zeros(2) - T))))                 # tag relative to the body, real
    assert truth == pytest.approx(0.0001003, abs=2e-7)
    res = nt.analyse([s0, s1, s0, s0], CFG, R_OK, tag_h=h)
    assert res['tru_than']['vi_tri_m'] - 0.0 >= truth                  # corrected ~0, eps covers
    assert res['than'][1]['quay_quanh_camera_m'] == pytest.approx(float(np.hypot(*((R - np.eye(2)) @ C))), rel=1e-6)


def test_nt7_plate_seated_higher_is_covered_by_the_scale():
    """Codex NT-7: plate 3 mm higher in one placement with a real 0.575 mm shift that projects to the same centre on
    z = h: same centre, scale up 3 mm / (H - h); eps_h = |s - 1| |p - C0| covers it."""
    H, h = 2.632, 0.12
    C = [-0.4802, -0.0329, H]
    s = summaries([(0.0, 0.0, 0.0)] * 4, [(0.0, 0.0, 0.0)] * 4, cam=C)
    s[1]['scale'] = (H - h) / (H - (h + 0.003))                       # projection p = C + a (P - C), a = (H-h)/(H-z)
    res = nt.analyse(s, CFG, R_OK, tag_h=h)
    real = 0.003 / (H - h) * math.hypot(C[0], C[1])                    # |P - p| = |1 - 1/a| |p - C|
    assert real == pytest.approx(0.000574832, abs=1e-9)
    assert res['vi_tri_m'] >= real * (1 - 1e-9) and res['tho']['vi_tri_m'] >= real * (1 - 1e-9)


def test_nt3_2_plate_seated_lower_uses_one_minus_one_over_r():
    """Codex NT3-2: plate 3 mm LOWER, same projected centre; real shift 0.574832 mm; |r - 1| alone gives 0.574146."""
    H, h = 2.632, 0.12
    C = [-0.4802, -0.0329, H]
    s = summaries([(0.0, 0.0, 0.0)] * 4, [(0.0, 0.0, 0.0)] * 4, cam=C)
    a = (H - h) / (H - (h - 0.003))
    s[1]['scale'] = a
    real = abs(1.0 - 1.0 / a) * math.hypot(C[0], C[1])
    assert real == pytest.approx(0.000574832, abs=1e-9)
    res = nt.analyse(s, CFG, R_OK, tag_h=h)
    assert res['vi_tri_m'] >= real


def test_nt3_1_scale_noise_widens_the_interval():
    """Codex NT3-1: a scale that looks equal to N0 still carries its statistical uncertainty."""
    H, h = 2.632, 0.12
    C = [-0.4802, -0.0329, H]
    s = summaries([(0.0, 0.0, 0.0)] * 4, [(0.0, 0.0, 0.0)] * 4, cam=C)
    for x in s:
        x['scale_u'] = 0.0005
    res = nt.analyse(s, CFG, R_OK, tag_h=h)
    assert res['than'][1]['eps_h_m'] > 0.00045                                   # ~0.1 % x 0.48 m
    assert res['than'][0]['eps_h_m'] == 0.0


def test_nt3_3_camera_height_change_with_a_lever_arm():
    """Codex NT3-3: camera centre moves by +-dC (x, y, z) in two clips; boxes really at 0.09 m projected at 0.12 m;
    body moved so the projected box centre stays put (t = 0); tag still at O. Real repeatability 0.098967 mm."""
    H0, h, zb = 2.632, 0.12, 0.09
    C0 = np.array([-0.4802, -0.0329])
    dC = np.array([0.003990645, 0.000273412, 0.000757397])
    q = [np.array([-0.11, 0.0]), np.array([0.11, 0.0])]
    a0 = (H0 - h) / (H0 - zb)
    summ, Ts = [], []
    for sgn in (0, 1, -1, 0):
        Ci, Hi = C0 + sgn * dC[:2], H0 + sgn * dC[2]
        ai = (Hi - h) / (Hi - zb)
        T = ((1 - a0) * C0 - (1 - ai) * Ci) / ai
        Ts.append(T)
        box = [(Ci + ai * (x + T - Ci)).tolist() for x in q]
        summ.append(dict(tag=[0.0, 0.0, 0.0], box=box, cam=[Ci[0], Ci[1], Hi], scale=1.0))
    real = float(np.hypot(*(Ts[1] - Ts[2])))
    assert real == pytest.approx(0.000098967, abs=2e-9)
    res = nt.analyse(summ, CFG, R_OK, tag_h=h)
    assert res['kiem_than'] and res['vi_tri_m'] >= real


def test_two_boxes_must_be_apart():
    s = summaries(STILL, PLATE_OFF)
    for x in s:
        x['box'] = [x['box'][0], [x['box'][0][0] + 0.03, x['box'][0][1]]]
    res = nt.analyse(s, CFG, R_OK)
    assert not res['kiem_than'] and any(w.startswith('vung_gan_nhau') for w in res['ly_do_khong_kiem_than'])


# ---------------------------------------------------------------- unit: clip summaries
def _rows(n=40, jump=None, corr=0.99, n_box=2, tag_seen=True, rms=0.5, res_sim=0.0005, outlier=None, peak2=0.5):
    rows = []
    for k in range(n):
        x = 0.1
        if jump is not None and k >= jump[0]:
            x += jump[1]
        if outlier is not None and k == outlier[0]:
            x += outlier[1]
        rows.append(dict(idx=k, t_s=k / 60.0, n_marks=20, rms_px=rms, gate_marks=rms <= 1.0, cam_x=0.0, cam_y=0.0,
                         cam_z=2.9, tag_seen=tag_seen, tag_x=x if tag_seen else math.nan, tag_y=0.2, tag_psi=0.3,
                         tag_res_sim_m=res_sim, tag_scale=1.0,
                         box=[(0.2 + i, 0.1, corr, 0.0, 0.0, peak2) for i in range(n_box)]))
    return rows


def test_still_clip_moving_clip_and_one_bad_frame():
    cfg = dict(CFG, **TEST_CFG)
    s = nt.clip_summary(_rows(), 2, cfg)
    assert s['ok'] and s['still'] and s['flags'] == [] and s['n_used'] == 40
    s = nt.clip_summary(_rows(jump=(30, 0.002)), 2, cfg)
    assert not s['ok'] and s['flags'] == ['clip_co_dong']
    s = nt.clip_summary(_rows(outlier=(20, 0.003)), 2, cfg)          # NT-9: one frame does not make a moving clip
    assert s['ok'] and s['tag_max_m'] == pytest.approx(0.003, abs=1e-9) and s['tag_p95_m'] < 1e-9


def test_frame_gates_and_frame_count():
    cfg = dict(CFG, **dict(TEST_CFG, min_frames_s1=10, min_frames_abs=3))
    for rows in (_rows(corr=0.5), _rows(tag_seen=False), _rows(n=5), _rows(rms=3.0), _rows(res_sim=0.003),
                 _rows(peak2=0.95)):
        assert nt.clip_summary(rows, 2, cfg)['flags'] == ['thieu_khung']
    assert nt.clip_summary(_rows(n=8), 2, cfg, stride=2)['ok']        # ceil(10 / 2) = 5 needed
    s = nt.clip_summary(_rows(n=30), 2, dict(cfg, trim_s=0.11))
    assert s['n_considered'] == 16                                    # 0.11 s: frames 0-6 and 23-29 dropped


def test_clip_r():
    cfg = dict(CFG, **TEST_CFG)
    ref_box, ref_tag = [[0.2, 0.1], [1.2, 0.1]], [0.1, 0.2]
    s = nt.rong_summary(_rows(jump=(0, 0.5)), ref_box, ref_tag, 2, cfg)  # tag 0.5 m away, boxes found
    assert s['dat'] and s['lech_vung_so_N0_m'] == [0.0, 0.0]
    s = nt.rong_summary(_rows(), ref_box, ref_tag, 2, cfg)              # tag still at its place: no proof
    assert not s['dat'] and s['n_ok'] == 0
    s = nt.rong_summary(_rows(tag_seen=False), ref_box, ref_tag, 2, cfg)  # NT2-3: tag not seen is no proof
    assert not s['dat']
    s = nt.rong_summary(_rows(jump=(0, 0.5), corr=0.3), ref_box, ref_tag, 2, cfg)   # boxes left with the plate
    assert not s['dat']
    s = nt.rong_summary(_rows(jump=(0, 0.5), peak2=0.97), ref_box, ref_tag, 2, cfg)  # ambiguous match
    assert not s['dat']


def test_parse_boxes():
    assert nt.parse_boxes('10,20,40,40; 100,200,30,32') == [(10, 20, 40, 40), (100, 200, 30, 32)]
    for bad in ('', '1,2,3', '1,2,4,40', '-1,2,40,40'):
        with pytest.raises(ValueError):
            nt.parse_boxes(bad)


# ---------------------------------------------------------------- unit: template matching
def texture(shape, seed=1, sigma=2.0):
    rng = np.random.default_rng(seed)
    t = cv2.GaussianBlur(rng.normal(0, 1, shape).astype(np.float32), (0, 0), sigma)
    t = (t - t.min()) / (t.max() - t.min()) * 200 + 25
    return t.astype(np.float32)


@pytest.mark.parametrize('dx,dy', [(0.37, -0.61), (-3.2, 2.45), (12.5, -7.1)])
def test_box_match_sub_pixel(dx, dy):
    """Error <= 0.05 px (about 0.04 px is the interpolation of the shifted test image itself)."""
    img = texture((240, 320))
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    moved = cv2.warpAffine(img, M, (320, 240), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    b = nt.Box(img.astype(np.uint8), (140, 100, 40, 40), 25)
    u, v, c, q = b.match(moved.astype(np.uint8))
    assert c > 0.95 and q < 0.9                                       # unambiguous peak
    assert u - b.u0 == pytest.approx(dx, abs=0.05) and v - b.v0 == pytest.approx(dy, abs=0.05)


def test_box_refuses_flat_or_edge():
    with pytest.raises(ValueError):
        nt.Box(np.full((200, 200), 100, np.uint8), (80, 80, 40, 40), 25)
    with pytest.raises(ValueError):
        nt.Box(texture((200, 200)).astype(np.uint8), (5, 80, 40, 40), 25)


# ---------------------------------------------------------------- end to end on rendered clips
def marks_dict():
    m = {}
    for group, side in zip(MARKS, SIDES):
        for i, x, y in group:
            m[i] = (x / 100, y / 100, side / 1000)
    return m


def floor_img(cam, res=0.002, xlim=(-2.6, 2.6), ylim=(-2.0, 2.0)):
    nx, ny = int((xlim[1] - xlim[0]) / res), int((ylim[1] - ylim[0]) / res)
    img = np.full((ny, nx), 200, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(cp.DICT)
    for i, (x, y, side) in marks_dict().items():
        s = int(round(side / res))
        tag = cv2.aruco.generateImageMarker(d, i, s, borderBits=1)
        cx, cy = (x - xlim[0]) / res, (ylim[1] - y) / res
        m = int(round(0.015 / res))
        r0, c0 = int(round(cy - s / 2)), int(round(cx - s / 2))
        img[r0 - m:r0 + s + m, c0 - m:c0 + s + m] = 250
        img[r0:r0 + s, c0:c0 + s] = np.where(tag > 128, 250, 20)
    u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    P = cam.pix2floor(np.column_stack([u.ravel(), v.ravel()]))
    mx = ((P[:, 0] - xlim[0]) / res).reshape(H, W).astype(np.float32)
    my = ((ylim[1] - P[:, 1]) / res).reshape(H, W).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderValue=128)


RES = 0.0005
BODY_TEX = (texture((int(BODY_W / RES), int(BODY_L / RES)), seed=7, sigma=16.0) * 0.6 + 10).astype(np.uint8)


def plate_tex():
    s = int(round(CAR_SIDE / RES))
    tag = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cp.DICT), 0, s, borderBits=1)
    tag = np.rot90(np.where(tag > 128, 250, 20).astype(np.uint8), -1)        # tag top edge -> +x of the plate
    m = int(round((PLATE - CAR_SIDE) / 2 / RES))
    p = np.full((s + 2 * m, s + 2 * m), 250, np.uint8)
    p[m:m + s, m:m + s] = tag
    return p


PLATE_TEX = plate_tex()


PLATE_AWAY = (0.0, -0.32, 0.0)          # clip R: plate lying next to the car (drawn at TAG_H, synthetic)


def render_car(frame, body, plate, half=320):
    """Car body at pose body = (x, y, psi), plate (tag centre) at plate = (dx, dy, dpsi) in the body frame, or None
    (plate off the car, clip R); both faces at TAG_H in this synthetic scene."""
    x, y, a = body
    uc, vc = CAM.project(np.array([[x, y, TAG_H]]))[0]
    u0, v0, u1, v1 = int(uc) - half, int(vc) - half, int(uc) + half, int(vc) + half
    u, v = np.meshgrid(np.arange(u0, u1, dtype=float), np.arange(v0, v1, dtype=float))
    P = CAM.pix2floor(np.column_stack([u.ravel(), v.ravel()]), TAG_H)
    c, s = math.cos(a), math.sin(a)
    xb = c * (P[:, 0] - x) + s * (P[:, 1] - y)
    yb = -s * (P[:, 0] - x) + c * (P[:, 1] - y)
    out = np.zeros(len(xb), np.float32)
    mb = cv2.remap(BODY_TEX, ((xb + BODY_L / 2) / RES).reshape(u.shape).astype(np.float32),
                   ((BODY_W / 2 - yb) / RES).reshape(u.shape).astype(np.float32), cv2.INTER_LINEAR).ravel()
    in_b = (np.abs(xb) <= BODY_L / 2) & (np.abs(yb) <= BODY_W / 2)
    out[in_b] = mb[in_b]
    # (a plate drawn outside the body, as in clip R, is drawn over the floor)
    msk = in_b
    if plate is not None:
        dx, dy, dp = plate
        cp_, sp = math.cos(dp), math.sin(dp)
        xp = cp_ * (xb - dx) + sp * (yb - dy)
        yp = -sp * (xb - dx) + cp_ * (yb - dy)
        mp = cv2.remap(PLATE_TEX, ((xp + PLATE / 2) / RES).reshape(u.shape).astype(np.float32),
                       ((PLATE / 2 - yp) / RES).reshape(u.shape).astype(np.float32), cv2.INTER_LINEAR).ravel()
        in_p = (np.abs(xp) <= PLATE / 2) & (np.abs(yp) <= PLATE / 2)
        out[in_p] = mp[in_p]
        msk = in_b | in_p
    roi = frame[v0:v1, u0:u1]
    m2 = msk.reshape(u.shape)
    roi[m2] = np.clip(out.reshape(u.shape)[m2], 0, 255).astype(np.uint8)


def write_video(path, frames):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), FPS, (W, H))
    assert vw.isOpened()
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    vw.release()


@pytest.fixture(scope='module')
def scene(tmp_path_factory):
    d = tmp_path_factory.mktemp('nt')
    floor = floor_img(CAM)
    write_video(d / 'H.avi', [floor] * 6)
    ses = cm.calibrate(str(d / 'H.avi'), marks_dict(), ZC, PLUMB, TAG_H, CAR_SIDE)
    with open(d / 'session.json', 'w', encoding='utf-8') as f:
        json.dump(cm._json(ses), f)
    return d, floor


def make_clip(d, floor, body, plate, name, n=8):
    f = floor.copy()
    render_car(f, body, plate)
    path = d / f'{name}.avi'
    write_video(path, [f] * n)
    return str(path)


def make_clips(d, floor, body, plate, tag):
    return [make_clip(d, floor, b, p, f'{tag}_N{j}') for j, (b, p) in enumerate(zip(body, plate))]


def boxes_px(body0, size=16):
    x, y, a = body0
    out = []
    for bx, by in BOX_BODY:
        px = np.array([x + math.cos(a) * bx - math.sin(a) * by, y + math.sin(a) * bx + math.cos(a) * by, TAG_H])
        u, v = CAM.project(px[None])[0]
        out.append(f'{int(round(u - size / 2))},{int(round(v - size / 2))},{size},{size}')
    return ';'.join(out)


def test_end_to_end_plate_moves_body_still(scene, tmp_path):
    d, floor = scene
    body = [(0.05, -0.03, math.radians(20.0))] * 5
    clips = make_clips(d, floor, body, PLATE_OFF, 'still')
    rong = make_clip(d, floor, body[0], PLATE_AWAY, 'still_R')
    out = tmp_path / 'o'
    code = nt.run_do(clips, str(d / 'session.json'), boxes_px(body[0]), str(out), rong=rong, cfg=TEST_CFG)
    r = json.load(open(out / 'nhac_tam.json', encoding='utf-8'))
    blk = json.load(open(out / 'c8_phan_nhac_tam.json', encoding='utf-8'))
    tp, ta = truth_rep(PLATE_OFF)
    assert code == 0, r['co']
    assert r['rong']['tom_tat']['dat'] and r['ket_qua']['kiem_than']
    assert r['ket_qua']['tru_than']['vi_tri_m'] == pytest.approx(tp, abs=1e-4)
    assert r['ket_qua']['tru_than']['goc_deg'] == pytest.approx(ta, abs=0.05)
    assert r['ket_qua']['than_dich_max_m'] < 1e-4
    assert blk['trang_thai'] == 'dat' and blk['n_lan_dat'] == 5
    assert blk['vi_tri_m'] == pytest.approx(tp, abs=1e-4) and blk['goc_deg'] == pytest.approx(ta, abs=0.05)
    assert len(blk['nguon']) == 7
    assert (out / 'frames_nhac_tam.csv').exists() and (out / 'bao_cao.txt').exists()


def test_end_to_end_body_pushed(scene, tmp_path):
    d, floor = scene
    body = [(0.05, -0.03, math.radians(20.0)), (0.0535, -0.03, math.radians(20.6)), (0.05, -0.027, math.radians(19.7)),
            (0.051, -0.031, math.radians(20.2)), (0.052, -0.029, math.radians(20.1))]
    clips = make_clips(d, floor, body, PLATE_OFF, 'push')
    rong = make_clip(d, floor, body[0], PLATE_AWAY, 'push_R')
    out = tmp_path / 'o'
    code = nt.run_do(clips, str(d / 'session.json'), boxes_px(body[0]), str(out), rong=rong, cfg=TEST_CFG)
    r = json.load(open(out / 'nhac_tam.json', encoding='utf-8'))
    tp, ta = truth_rep(PLATE_OFF)
    assert code == 0, r['co']
    k = r['ket_qua']
    assert k['tru_than']['vi_tri_m'] == pytest.approx(tp, abs=1.5e-4)
    assert k['tru_than']['goc_deg'] == pytest.approx(ta, abs=0.05)
    assert k['than_dich_max_m'] == pytest.approx(0.0035, abs=2e-4)
    assert k['than_dich_max_deg'] == pytest.approx(0.6, abs=0.05)
    assert k['tho']['vi_tri_m'] > tp and k['vi_tri_m'] == k['tru_than']['vi_tri_m']


def test_end_to_end_without_clip_r_is_limited(scene, tmp_path):
    d, floor = scene
    body = [(0.05, -0.03, math.radians(20.0))] * 4
    clips = make_clips(d, floor, body, PLATE_OFF[:4], 'nor')
    out = tmp_path / 'o'
    code = nt.run_do(clips, str(d / 'session.json'), boxes_px(body[0]), str(out), cfg=TEST_CFG)
    blk = json.load(open(out / 'c8_phan_nhac_tam.json', encoding='utf-8'))
    assert code == 1 and blk['trang_thai'] == 'dat_gioi_han' and not blk['kiem_than']
    assert blk['vi_tri_m'] == blk['vi_tri_tho_m']


def test_blocking_error_writes_no_block(scene, tmp_path):
    d, floor = scene
    body = [(0.05, -0.03, math.radians(20.0))] * 4
    clips = make_clips(d, floor, body, PLATE_OFF[:4], 'blk')
    out = tmp_path / 'o'
    code = nt.run_do(clips, str(d / 'session.json'), boxes_px(body[0]), str(out),
                     cfg=dict(TEST_CFG, min_frames_s1=100))
    assert code == 2 and not (out / 'c8_phan_nhac_tam.json').exists() and (out / 'nhac_tam.json').exists()


def test_run_do_refusals(scene, tmp_path):
    d, _ = scene
    with pytest.raises(ValueError):
        nt.run_do(['a', 'b', 'c'], str(d / 'session.json'), '1,1,20,20', str(tmp_path / 'x'))
    (tmp_path / 'full').mkdir()
    (tmp_path / 'full' / 'f.txt').write_text('x')
    with pytest.raises(ValueError):
        nt.run_do(['a'] * 4, str(d / 'session.json'), '1,1,20,20', str(tmp_path / 'full'))


def test_cli_last_line(scene, tmp_path, capsys):
    d, _ = scene
    code = nt.main(['do', '--clip', 'a', 'b', '--session', str(d / 'session.json'), '--vung-than', '1,1,20,20',
                    '--out', str(tmp_path / 'c')])
    assert code == 2 and capsys.readouterr().out.strip().splitlines()[-1] == '#ket_qua,ma=2'


def test_end_to_end_stride_above_3_is_a_quick_look(scene, tmp_path):
    d, floor = scene
    body = [(0.05, -0.03, math.radians(20.0))] * 4
    clips = [make_clip(d, floor, b, p, f'qs_N{j}', n=16) for j, (b, p) in enumerate(zip(body, PLATE_OFF[:4]))]
    out = tmp_path / 'o'
    code = nt.run_do(clips, str(d / 'session.json'), boxes_px(body[0]), str(out), stride=4, cfg=TEST_CFG)
    r = json.load(open(out / 'nhac_tam.json', encoding='utf-8'))
    assert code == 2 and 'stride_xem_nhanh' in r['co'] and not (out / 'c8_phan_nhac_tam.json').exists()


def test_kiem_one_clip_and_rejected_list(scene, tmp_path):
    d, floor = scene
    b0 = (0.05, -0.03, math.radians(20.0))
    n0 = make_clip(d, floor, b0, PLATE_OFF[0], 'k_N0')
    r = make_clip(d, floor, b0, PLATE_AWAY, 'k_R')
    rbad = make_clip(d, floor, b0, PLATE_OFF[1], 'k_Rbad')            # plate still on the car: not a clip R
    ok, txt = nt.run_kiem(n0, str(d / 'session.json'), boxes_px(b0), n0, str(tmp_path / 'a'), stride=1, cfg=TEST_CFG)
    assert ok and txt.endswith('ĐƯỢC')
    ok, _ = nt.run_kiem(r, str(d / 'session.json'), boxes_px(b0), n0, str(tmp_path / 'b'), rong=True, stride=1,
                        cfg=TEST_CFG)
    assert ok and (tmp_path / 'b' / 'khung_R.png').exists()
    ok, txt = nt.run_kiem(rbad, str(d / 'session.json'), boxes_px(b0), n0, str(tmp_path / 'c'), rong=True, stride=1,
                          cfg=TEST_CFG)
    assert not ok and 'QUAY LẠI' in txt
    clips = [n0] + [make_clip(d, floor, b0, p, f'k_N{j}') for j, p in enumerate(PLATE_OFF[1:4], 1)]
    out = tmp_path / 'o'
    code = nt.run_do(clips, str(d / 'session.json'), boxes_px(b0), str(out), rong=r, cfg=TEST_CFG,
                     loai=[f'{rbad}=khong phai clip R'])
    j = json.load(open(out / 'nhac_tam.json', encoding='utf-8'))
    assert code == 0 and j['clip_bi_loai'][0]['file'] == 'k_Rbad.avi' and len(j['clip_bi_loai'][0]['sha256']) == 64
    with pytest.raises(ValueError):
        nt.parse_rejected(['khong_co_dau_bang'])
