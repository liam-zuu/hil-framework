"""Tests for tools/c7_measure.py on SYNTHETIC clips rendered here from a known camera (NOT a measurement).
Floor marks ID 1-24 at the print-CSV centres, check tags ID 25-33, car tag ID 0 on a plate at height h."""
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
from test_c7_clip_probe import MARKS, SIDES, W, H, FPS, truth_camera  # noqa: E402

CHECK = [(25, -100, 0), (26, -50, 0), (27, 0, 0), (28, 50, 0), (29, 100, 0), (30, -70, 50), (31, 70, 50),
         (32, -70, -50), (33, 70, -50)]
TAG_H = 0.12
CAR_SIDE = 0.15
CAR_L, CAR_W = 0.34, 0.29
OFFSET = (0.003, -0.002, math.radians(0.8))       # tag centre - wheel centre (body), tag heading - body


def marks_dict(check=True):
    m = {}
    for group, side in zip(MARKS, SIDES):
        for i, x, y in group:
            m[i] = (x / 100, y / 100, side / 1000)
    if check:
        for i, x, y in CHECK:
            m[i] = (x / 100, y / 100, 0.1)
    return m


def floor_texture(check, res=0.002, xlim=(-2.6, 2.6), ylim=(-2.0, 2.0)):
    nx, ny = int((xlim[1] - xlim[0]) / res), int((ylim[1] - ylim[0]) / res)
    img = np.full((ny, nx), 200, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(cp.DICT)
    for i, (x, y, side) in marks_dict(check).items():
        s = int(round(side / res))
        tag = cv2.aruco.generateImageMarker(d, i, s, borderBits=1)
        cx, cy = (x - xlim[0]) / res, (ylim[1] - y) / res
        m = int(round(0.015 / res))
        r0, c0 = int(round(cy - s / 2)), int(round(cx - s / 2))
        img[r0 - m:r0 + s + m, c0 - m:c0 + s + m] = 250
        img[r0:r0 + s, c0:c0 + s] = np.where(tag > 128, 250, 20)
    return img, res, xlim, ylim


def render_floor(cam, tex):
    img, res, xlim, ylim = tex
    u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    P = cam.pix2floor(np.column_stack([u.ravel(), v.ravel()]))
    mx = ((P[:, 0] - xlim[0]) / res).reshape(H, W).astype(np.float32)
    my = ((ylim[1] - P[:, 1]) / res).reshape(H, W).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderValue=128)


def car_texture(res=0.0005):
    nx, ny = int(round(CAR_L / res)), int(round(CAR_W / res))
    tex = np.full((ny, nx), 60, np.uint8)
    s = int(round(CAR_SIDE / res))
    tag = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cp.DICT), 0, s, borderBits=1)
    tag = np.where(tag > 128, 250, 20).astype(np.uint8)
    # body frame: column = x forward, row = -y; tag top edge (image row 0) towards +x forward
    tag = np.rot90(tag, -1)                                  # tag top -> +x (columns)
    m = int(round(0.015 / res))
    paper = np.full((s + 2 * m, s + 2 * m), 250, np.uint8)
    paper[m:m + s, m:m + s] = tag
    rot = cv2.getRotationMatrix2D(((s + 2 * m) / 2 - 0.5, (s + 2 * m) / 2 - 0.5), math.degrees(OFFSET[2]), 1.0)
    paper = cv2.warpAffine(paper, rot, paper.shape[::-1], flags=cv2.INTER_LINEAR, borderValue=60)
    cx, cy = (CAR_L / 2 + OFFSET[0]) / res, (CAR_W / 2 - OFFSET[1]) / res
    r0, c0 = int(round(cy - paper.shape[0] / 2)), int(round(cx - paper.shape[1] / 2))
    tex[r0:r0 + paper.shape[0], c0:c0 + paper.shape[1]] = paper
    return tex, res


def render_car(frame, cam, car, pose, half=260):
    tex, res = car
    x, y, psi = pose
    uc, vc = cam.project(np.array([[x, y, TAG_H]]))[0]
    u0, v0 = max(0, int(uc) - half), max(0, int(vc) - half)
    u1, v1 = min(W, int(uc) + half), min(H, int(vc) + half)
    u, v = np.meshgrid(np.arange(u0, u1, dtype=float), np.arange(v0, v1, dtype=float))
    P = cam.pix2floor(np.column_stack([u.ravel(), v.ravel()]), TAG_H)
    c, s = math.cos(psi), math.sin(psi)
    xb = c * (P[:, 0] - x) + s * (P[:, 1] - y)
    yb = -s * (P[:, 0] - x) + c * (P[:, 1] - y)
    mx = ((xb + CAR_L / 2) / res).reshape(u.shape).astype(np.float32)
    my = ((CAR_W / 2 - yb) / res).reshape(u.shape).astype(np.float32)
    inside = (np.abs(xb) <= CAR_L / 2) & (np.abs(yb) <= CAR_W / 2)
    patch = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderValue=0)
    roi = frame[v0:v1, u0:u1]
    roi[inside.reshape(u.shape)] = patch[inside.reshape(u.shape)]


def write_video(path, frames):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), FPS, (W, H))
    assert vw.isOpened()
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    vw.release()


CAM = truth_camera()
PLUMB = tuple(CAM.centre[:2])
ZC = float(CAM.centre[2])


@pytest.fixture(scope='module')
def session(tmp_path_factory):
    d = tmp_path_factory.mktemp('h')
    f = render_floor(CAM, floor_texture(check=True))
    write_video(d / 'H.avi', [f] * 6)
    return cm.calibrate(str(d / 'H.avi'), marks_dict(check=True), ZC, PLUMB, TAG_H, CAR_SIDE)


def poses_line(n=24):
    return [(-0.6 + 1.2 * k / (n - 1), 0.3 - 0.1 * k / (n - 1), math.radians(30.0 + 20.0 * k / (n - 1)))
            for k in range(n)]


@pytest.fixture(scope='module')
def run_clip(tmp_path_factory, session):
    d = tmp_path_factory.mktemp('r')
    tex = floor_texture(check=False)
    base = render_floor(CAM, tex)
    rx, ry, rz = 0.002, -0.0015, 0.001                       # one bumped frame: camera turned ~0.15 deg
    R = cv2.Rodrigues(np.array([rx, ry, rz]))[0] @ cv2.Rodrigues(CAM.rvec)[0]
    C = CAM.centre
    bumped = cp.Camera(CAM.model, CAM.f, CAM.cx, CAM.cy, CAM.k, cv2.Rodrigues(R)[0].ravel(), -R @ C)
    base_b = render_floor(bumped, tex)
    car = car_texture()
    poses = poses_line()
    frames = []
    for k, p in enumerate(poses):
        cam = bumped if k == 12 else CAM
        f = (base_b if k == 12 else base).copy()
        render_car(f, cam, car, p)
        frames.append(f)
    write_video(d / 'run.avi', frames)
    return str(d / 'run.avi'), poses


def test_refine_edges_has_no_inward_bias():
    s = 60
    tag = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cp.DICT), 7, s, borderBits=1)
    img = np.full((200, 200), 240, np.uint8)
    img[70:70 + s, 70:70 + s] = np.where(tag > 128, 240, 20)
    img = cv2.GaussianBlur(img, (0, 0), 1.5)
    truth = np.array([[69.5, 69.5], [129.5, 69.5], [129.5, 129.5], [69.5, 129.5]])
    d = cm.detect_refined(cm.make_detector(), img)
    assert 7 in d
    err = d[7] - truth
    assert np.max(np.abs(err)) < 0.1
    assert abs(np.mean(np.linalg.norm(d[7] - truth.mean(0), axis=1)) - np.mean(np.linalg.norm(truth - truth.mean(0), axis=1))) < 0.05


def test_calibration_from_floor_marks(session):
    s = session
    assert s['chosen'] in ('B1', 'B2')
    assert s['rms_px_H'] < 0.5
    assert abs(s['zc_fit_m'] - ZC) < 0.03
    assert len(s['check_tags_err_mm']) == 9
    assert max(s['check_tags_err_mm'].values()) < 3.0
    assert abs(s['kappa'] - TAG_H / ZC) < 1e-9
    assert s['tag_offset_src'].startswith('chua do')


def test_track_recovers_wheel_centre_with_parallax_and_offset(session, run_clip):
    video, poses = run_clip
    rows, info = cm.track(video, session, tag_offset=OFFSET)
    assert info['frames'] == len(poses)
    assert all(r['n_marks'] >= 20 for r in rows)
    errs = [math.hypot(r['x'] - p[0], r['y'] - p[1]) for r, p in zip(rows, poses) if r['car_seen']]
    herr = [abs(cm.cp_wrap(r['psi'] - p[2])) for r, p in zip(rows, poses) if r['car_seen']]
    assert len(errs) == len(poses)
    assert max(errs) < 0.003, max(errs)
    assert math.degrees(max(herr)) < 0.5                              # 0.30 deg seen on this render
    assert rows[12]['shift_px'] > 2.0 > rows[11]['shift_px']          # the bumped frame is seen ...
    assert math.hypot(rows[12]['x'] - poses[12][0], rows[12]['y'] - poses[12][1]) < 0.003   # ... and compensated
    assert sum(r['valid'] for r in rows) >= len(poses) - 2


def test_without_parallax_the_error_is_large(session, run_clip):
    video, poses = run_clip
    s = dict(session, kappa=0.0)
    rows, _ = cm.track(video, s, tag_offset=OFFSET, max_frames=3)
    errs = [math.hypot(r['x'] - p[0], r['y'] - p[1]) for r, p in zip(rows, poses)]
    assert max(errs) > 0.02


def test_frames_csv_round_trip(tmp_path, session, run_clip):
    rows, _ = cm.track(run_clip[0], session, tag_offset=OFFSET, max_frames=8)
    cm.write_frames(tmp_path / 'f.csv', rows)
    back = cm.read_frames(tmp_path / 'f.csv')
    assert len(back) == 8 and back[3]['valid'] == rows[3]['valid']
    assert abs(back[3]['x'] - rows[3]['x']) < 1e-6


def test_gates_drop_a_jumping_frame():
    rows = []
    for k in range(20):
        rows.append(dict(t_s=k / 60, n_marks=20, rms_px=0.3, car_seen=True, res_m=0.0003, scale=1.0,
                         x=0.01 * k + (0.05 if k == 10 else 0.0), y=0.0, psi=0.0))
    rows[5]['n_marks'] = 6
    rows[7]['res_m'] = 0.004
    rows[15]['scale'] = 1.02
    cm.apply_gates(rows, rms_h=0.3)
    bad = {k: r['why'] for k, r in enumerate(rows) if not r['valid']}
    assert set(bad) == {5, 7, 10, 15}
    assert bad[10] == ['nhay_vi_tri'] and bad[5] == ['moc_it'] and bad[7] == ['xe_du_thua'] and bad[15] == ['ti_le']
    assert all(np.isfinite(r['speed']) for r in rows if r['valid'])
