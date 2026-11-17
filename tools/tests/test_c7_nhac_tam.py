"""Tests for tools/c7_nhac_tam.py (plate lift test of C8, design 2026-11-18_Thiet_ke_c7_nhac_tam_v1.md section 4).
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
PLATE = 0.18                       # tag 0.15 + white margin 0.015 each side
BOX_BODY = [(0.15, 0.125), (-0.15, -0.125)]          # tape centres in the body frame (outside the plate)
CAM = truth_camera()
PLUMB = tuple(CAM.centre[:2])
ZC = float(CAM.centre[2])
TEST_CFG = dict(trim_s=0.0, min_frames=3, min_frac=0.5)


# ---------------------------------------------------------------- unit: analysis on synthetic summaries
def rot(a):
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def summaries(body, plate, boxes_on_plate=False, bend=None):
    """body[j] = (x, y, psi) body pose, plate[j] = (dx, dy, dpsi) plate (tag centre) in the body frame."""
    out = []
    bb = [np.array([0.13, 0.11]), np.array([-0.13, -0.11]), np.array([0.13, -0.11])]
    for j, ((x, y, a), (dx, dy, dp)) in enumerate(zip(body, plate)):
        R, t = rot(a), np.array([x, y])
        tag = R @ np.array([dx, dy]) + t
        if boxes_on_plate:            # tape stuck on the plate: it moves with the tag
            Rp = rot(a + dp)
            pts = [Rp @ b + tag for b in bb]
        else:
            pts = [R @ b + t for b in bb]
        if bend is not None and j == bend[0]:
            pts[0] = pts[0] + np.array([bend[1], 0.0])
        out.append(dict(ok=True, tag=[tag[0], tag[1], nt.wrap(a + dp)], box=[p.tolist() for p in pts]))
    return out


PLATE_OFF = [(0.0, 0.0, 0.0), (0.0008, 0.0, math.radians(0.3)), (-0.0003, 0.0005, math.radians(-0.1)),
             (0.0, -0.0004, math.radians(0.15))]


def truth_rep(plate):
    best_p = max(math.hypot(a[0] - b[0], a[1] - b[1]) for a in plate for b in plate)
    best_a = max(abs(math.degrees(a[2] - b[2])) for a in plate for b in plate)
    return best_p, best_a


def test_body_motion_removed_and_value_is_the_largest():
    body = [(0.0, 0.0, 0.0), (0.002, -0.001, math.radians(0.5)), (0.0015, 0.0, math.radians(-0.4)),
            (0.0, 0.002, math.radians(0.2))]
    res, flags = nt.analyse(summaries(body, PLATE_OFF), dict(nt.DEFAULTS))
    tp, ta = truth_rep(PLATE_OFF)
    assert res['kiem_than']
    assert res['tru_than']['vi_tri_m'] == pytest.approx(tp, abs=1e-9)
    assert res['tru_than']['goc_deg'] == pytest.approx(ta, abs=1e-9)
    assert res['tho']['vi_tri_m'] > tp and res['tho']['goc_deg'] > ta
    assert res['vi_tri_m'] == res['tho']['vi_tri_m'] and res['goc_deg'] == res['tho']['goc_deg']
    assert 'than_dich_nhieu' in flags                        # 2.2 mm, 0.5 deg pushed: redo at the floor
    assert res['than_dich_max_m'] == pytest.approx(max(math.hypot(b[0], b[1]) for b in body), abs=1e-9)
    assert res['than_dich_max_deg'] == pytest.approx(0.5, abs=1e-9)


def test_still_body_gives_the_plate_repeatability_without_flags():
    body = [(0.3, -0.2, math.radians(40.0))] * 4
    res, flags = nt.analyse(summaries(body, PLATE_OFF), dict(nt.DEFAULTS))
    tp, ta = truth_rep(PLATE_OFF)
    assert flags == []
    assert res['vi_tri_m'] == pytest.approx(tp, abs=1e-9) and res['goc_deg'] == pytest.approx(ta, abs=1e-9)
    assert res['than_dich_max_m'] == pytest.approx(0.0, abs=1e-12)


def test_tape_on_the_plate_cannot_hide_the_repeatability():
    """Design 2.4: boxes moving with the plate make the body-removed value ~0; the value keeps the raw one."""
    body = [(0.0, 0.0, 0.0)] * 4
    res, _ = nt.analyse(summaries(body, PLATE_OFF, boxes_on_plate=True), dict(nt.DEFAULTS))
    tp, ta = truth_rep(PLATE_OFF)
    assert res['tru_than']['vi_tri_m'] < 1e-6 and res['tru_than']['goc_deg'] < 1e-4
    assert res['vi_tri_m'] == pytest.approx(tp, abs=1e-9) and res['goc_deg'] == pytest.approx(ta, abs=1e-9)


def test_non_rigid_box_turns_the_body_check_off():
    body = [(0.0, 0.0, 0.0)] * 4
    res, flags = nt.analyse(summaries(body, PLATE_OFF, bend=(2, 0.0015)), dict(nt.DEFAULTS))
    assert not res['kiem_than'] and 'khong_kiem_than' in flags
    assert res['tru_than'] is None and res['vi_tri_m'] == res['tho']['vi_tri_m']
    assert any(w.startswith('N2_du_khop') for w in res['ly_do_khong_kiem_than'])


def test_one_box_is_not_a_body_check():
    s = summaries([(0.0, 0.0, 0.0)] * 4, PLATE_OFF)
    for x in s:
        x['box'] = x['box'][:1]
    res, flags = nt.analyse(s, dict(nt.DEFAULTS))
    assert not res['kiem_than'] and res['ly_do_khong_kiem_than'] == ['it_hon_2_vung'] and 'khong_kiem_than' in flags


def test_angles_wrap_around_180():
    poses = [[0, 0, math.radians(179.9)], [0, 0, math.radians(-179.9)], [0, 0, math.pi], [0, 0, -math.pi]]
    assert nt.pairwise(poses)['goc_deg'] == pytest.approx(0.2, abs=1e-9)
    assert math.degrees(nt.circ_median([math.radians(179.0), math.radians(-179.0), math.radians(179.5)])) \
        == pytest.approx(179.5, abs=1e-9)


# ---------------------------------------------------------------- unit: clip summary
def _rows(n=20, jitter=0.0, jump=None, corr=0.99, n_box=2, tag_seen=True):
    rows = []
    for k in range(n):
        x = 0.1 + (jitter * (-1) ** k)
        if jump is not None and k >= jump[0]:
            x += jump[1]
        rows.append(dict(idx=k, t_s=k / 60.0, n_marks=20, rms_px=0.5, tag_seen=tag_seen, tag_x=x, tag_y=0.2,
                         tag_psi=0.3, box=[(0.2 + i, 0.1, corr, 0.0, 0.0) for i in range(n_box)]))
    return rows


def test_still_clip_and_moving_clip():
    cfg = dict(nt.DEFAULTS, **TEST_CFG)
    s = nt.clip_summary(_rows(jitter=0.00002), 2, cfg)
    assert s['ok'] and s['still'] and s['flags'] == [] and s['n_used'] == 20
    s = nt.clip_summary(_rows(jump=(15, 0.002)), 2, cfg)
    assert s['ok'] and not s['still'] and s['flags'] == ['clip_co_dong']


def test_too_few_frames_or_low_correlation():
    cfg = dict(nt.DEFAULTS, **dict(TEST_CFG, min_frames=10))
    assert nt.clip_summary(_rows(corr=0.5), 2, cfg)['flags'] == ['thieu_khung']
    assert nt.clip_summary(_rows(tag_seen=False), 2, cfg)['flags'] == ['thieu_khung']
    assert nt.clip_summary(_rows(n=5), 2, cfg)['flags'] == ['thieu_khung']
    s = nt.clip_summary(_rows(n=30), 2, dict(nt.DEFAULTS, **dict(TEST_CFG, trim_s=0.11, min_frames=10)))
    assert s['n_after_trim'] == 16                           # 0.11 s: frames 0-6 and 23-29 dropped (60 fps)


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
    img = texture((240, 320))
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    moved = cv2.warpAffine(img, M, (320, 240), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    b = nt.Box(img.astype(np.uint8), (140, 100, 40, 40), 25)
    u, v, c = b.match(moved.astype(np.uint8))
    assert c > 0.95
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


def render_car(frame, body, plate, half=200):
    """Car body at pose body = (x, y, psi), plate (tag centre) at plate = (dx, dy, dpsi) in the body frame; both
    faces at TAG_H (the tape on the body and the tag at the same height in this synthetic scene)."""
    x, y, a = body
    uc, vc = CAM.project(np.array([[x, y, TAG_H]]))[0]
    u0, v0, u1, v1 = int(uc) - half, int(vc) - half, int(uc) + half, int(vc) + half
    u, v = np.meshgrid(np.arange(u0, u1, dtype=float), np.arange(v0, v1, dtype=float))
    P = CAM.pix2floor(np.column_stack([u.ravel(), v.ravel()]), TAG_H)
    c, s = math.cos(a), math.sin(a)
    xb = c * (P[:, 0] - x) + s * (P[:, 1] - y)
    yb = -s * (P[:, 0] - x) + c * (P[:, 1] - y)
    dx, dy, dp = plate
    cp_, sp = math.cos(dp), math.sin(dp)
    xp = cp_ * (xb - dx) + sp * (yb - dy)
    yp = -sp * (xb - dx) + cp_ * (yb - dy)
    out = np.zeros(len(xb), np.float32)
    mb = cv2.remap(BODY_TEX, ((xb + BODY_L / 2) / RES).reshape(u.shape).astype(np.float32),
                   ((BODY_W / 2 - yb) / RES).reshape(u.shape).astype(np.float32), cv2.INTER_LINEAR).ravel()
    mp = cv2.remap(PLATE_TEX, ((xp + PLATE / 2) / RES).reshape(u.shape).astype(np.float32),
                   ((PLATE / 2 - yp) / RES).reshape(u.shape).astype(np.float32), cv2.INTER_LINEAR).ravel()
    in_b = (np.abs(xb) <= BODY_L / 2) & (np.abs(yb) <= BODY_W / 2)
    in_p = (np.abs(xp) <= PLATE / 2) & (np.abs(yp) <= PLATE / 2)
    out[in_b] = mb[in_b]
    out[in_p] = mp[in_p]
    roi = frame[v0:v1, u0:u1]
    msk = (in_b | in_p).reshape(u.shape)
    roi[msk] = np.clip(out.reshape(u.shape)[msk], 0, 255).astype(np.uint8)


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


def make_clips(d, floor, body, plate, tag, n=8):
    paths = []
    for j, (b, p) in enumerate(zip(body, plate)):
        f = floor.copy()
        render_car(f, b, p)
        path = d / f'{tag}_N{j}.avi'
        write_video(path, [f] * n)
        paths.append(str(path))
    return paths


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
    body = [(0.05, -0.03, math.radians(20.0))] * 4
    clips = make_clips(d, floor, body, PLATE_OFF, 'still')
    out = tmp_path / 'o'
    code = nt.run_do(clips, str(d / 'session.json'), boxes_px(body[0]), str(out), cfg=TEST_CFG)
    r = json.load(open(out / 'nhac_tam.json', encoding='utf-8'))
    blk = json.load(open(out / 'c8_phan_nhac_tam.json', encoding='utf-8'))
    tp, ta = truth_rep(PLATE_OFF)
    assert code == 0, r['co']
    assert r['ket_qua']['kiem_than']
    assert r['ket_qua']['tru_than']['vi_tri_m'] == pytest.approx(tp, abs=1e-4)
    assert r['ket_qua']['tru_than']['goc_deg'] == pytest.approx(ta, abs=0.05)
    assert r['ket_qua']['than_dich_max_m'] < 1e-4
    assert blk['vi_tri_m'] >= blk['vi_tri_tru_than_m'] and blk['n_lan_dat'] == 4
    assert blk['vi_tri_m'] == pytest.approx(tp, abs=1e-4) and blk['goc_deg'] == pytest.approx(ta, abs=0.05)
    assert len(blk['nguon']) == 5
    assert (out / 'frames_nhac_tam.csv').exists() and (out / 'bao_cao.txt').exists()


def test_end_to_end_body_pushed(scene, tmp_path):
    d, floor = scene
    body = [(0.05, -0.03, math.radians(20.0)), (0.0535, -0.03, math.radians(20.6)), (0.05, -0.027, math.radians(19.7)),
            (0.051, -0.031, math.radians(20.2))]
    clips = make_clips(d, floor, body, PLATE_OFF, 'push')
    out = tmp_path / 'o'
    code = nt.run_do(clips, str(d / 'session.json'), boxes_px(body[0]), str(out), cfg=TEST_CFG)
    r = json.load(open(out / 'nhac_tam.json', encoding='utf-8'))
    tp, ta = truth_rep(PLATE_OFF)
    assert code == 1 and 'than_dich_nhieu' in r['co']
    k = r['ket_qua']
    assert k['tru_than']['vi_tri_m'] == pytest.approx(tp, abs=1.5e-4)
    assert k['tru_than']['goc_deg'] == pytest.approx(ta, abs=0.05)
    assert k['than_dich_max_m'] == pytest.approx(0.0035, abs=2e-4)
    assert k['than_dich_max_deg'] == pytest.approx(0.6, abs=0.05)
    assert k['vi_tri_m'] == k['tho']['vi_tri_m'] > tp


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
