"""Tests for tools/c7_clip_probe.py on SYNTHETIC clips (rendered here from a known camera; NOT a measurement).
The marks are those of the print CSVs (ID 1-24 at the nominal centres), written to a temp CSV."""
import csv
import hashlib
import math
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import c7_clip_probe as cp  # noqa: E402

W, H, FPS = 1920, 1080, 60.0
MARKS = ([(1, -140, 90), (2, -70, 90), (3, 0, 90), (4, 70, 90), (5, 140, 90), (6, -140, 0), (7, 140, 0),
          (8, -140, -90), (9, -70, -90), (10, 0, -90), (11, 70, -90), (12, 140, -90)],
         [(13, -195, 145), (14, -97.5, 145), (15, 0, 145), (16, 97.5, 145), (17, 195, 145), (18, -195, 0),
          (19, 195, 0), (20, -195, -145), (21, -97.5, -145), (22, 0, -145), (23, 97.5, -145), (24, 195, -145)])
SIDES = (100.0, 150.0)
TRUTH_K = (-0.05, 0.005)


def truth_camera(model='B2', k=TRUTH_K, tilt_deg=(1.0, -0.7, 0.5), C=(0.03, -0.02, 2.90), f=900.0):
    """World -> camera for a phone looking down: image x along +x, image y along -y."""
    R0 = np.diag([1.0, -1.0, -1.0])
    rx, ry, rz = (math.radians(a) for a in tilt_deg)
    Rt = cv2.Rodrigues(np.array([rx, ry, rz]))[0]
    R = Rt @ R0
    t = -R @ np.array(C)
    return cp.Camera(model, f, W / 2, H / 2, k, cv2.Rodrigues(R)[0].ravel(), t)


def floor_texture(res=0.002, xlim=(-2.6, 2.6), ylim=(-2.0, 2.0)):
    nx, ny = int((xlim[1] - xlim[0]) / res), int((ylim[1] - ylim[0]) / res)
    img = np.full((ny, nx), 235, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(cp.DICT)
    for group, side in zip(MARKS, SIDES):
        for i, x, y in group:
            s = int(round(side / 1000 / res))
            tag = cv2.aruco.generateImageMarker(d, i, s, borderBits=1)
            cx, cy = (x / 100 - xlim[0]) / res, (ylim[1] - y / 100) / res          # row 0 is +y
            m = int(round(0.015 / res))                                           # white margin 15 mm
            r0, c0 = int(round(cy - s / 2)), int(round(cx - s / 2))
            img[r0 - m:r0 + s + m, c0 - m:c0 + s + m] = 250
            img[r0:r0 + s, c0:c0 + s] = np.where(tag > 128, 250, 20)
    return img, res, xlim, ylim


def render_maps(cam, tex):
    img, res, xlim, ylim = tex
    u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    P = cam.pix2floor(np.column_stack([u.ravel(), v.ravel()]))
    mx = ((P[:, 0] - xlim[0]) / res).reshape(H, W).astype(np.float32)
    my = ((ylim[1] - P[:, 1]) / res).reshape(H, W).astype(np.float32)
    return mx, my


def write_clip(path, cam, n=30, shift=None):
    tex = floor_texture()
    mx, my = render_maps(cam, tex)
    frame = cv2.remap(tex[0], mx, my, cv2.INTER_AREA, borderValue=128)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    assert vw.isOpened()
    for k in range(n):
        f = frame
        if shift and k in shift:
            f = cv2.warpAffine(frame, np.float32([[1, 0, shift[k]], [0, 1, 0]]), (W, H), borderValue=128)
        vw.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
    vw.release()


@pytest.fixture(scope='module')
def marks_csv(tmp_path_factory):
    p = tmp_path_factory.mktemp('m') / 'marks.csv'
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['id', 'vai_tro', 'x_cm', 'y_cm', 'canh_den_mm', 'dict'])
        w.writerow([0, 'noc_xe', '', '', 150, 'DICT_4X4_50'])
        for group, side in zip(MARKS, SIDES):
            for i, x, y in group:
                w.writerow([i, 'moc', x, y, side, 'DICT_4X4_50'])
    return str(p)


@pytest.fixture(scope='module')
def still_clip(tmp_path_factory):
    p = tmp_path_factory.mktemp('v') / 'still.mp4'
    write_clip(p, truth_camera())
    return str(p)


@pytest.mark.parametrize('model,k', [('B1', [-0.06]), ('B2', [-0.05, 0.005]), ('KB2', [0.02, -0.01])])
def test_project_and_pix2floor_are_inverse(model, k):
    cam = truth_camera(model, k)
    P = np.array([[-1.9, 1.4, 0], [0, 0, 0], [1.2, -0.7, 0], [0.4, 1.1, 0]])
    uv = cam.project(P)
    assert np.allclose(cam.pix2floor(uv), P[:, :2], atol=1e-6)


def test_load_marks_skips_rows_without_centre(marks_csv):
    m = cp.load_marks([marks_csv])
    assert 0 not in m and len(m) == 24 and m[13] == pytest.approx((-1.95, 1.45, 0.15))


def test_corner_order_matches_opencv_on_a_rendered_tag():
    tex = floor_texture()
    cam = truth_camera(k=(0.0, 0.0), tilt_deg=(0, 0, 0), C=(0, 0, 2.6))
    mx, my = render_maps(cam, tex)
    img = cv2.remap(tex[0], mx, my, cv2.INTER_AREA, borderValue=128)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cp.DICT), cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(img)
    got = {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)}
    exp = cam.project(np.column_stack([cp.mark_corners_floor(0.70, 0.90, 0.10), np.zeros(4)]))
    assert np.max(np.abs(got[4] - exp)) < 1.5                          # same corner order, within detection error


def test_probe_on_a_still_synthetic_clip(still_clip, marks_csv, tmp_path):
    out = tmp_path / 'probe'
    assert cp.main(['--video', still_clip, '--markers', marks_csv, '--zc', '2.90', '--out', str(out)]) == 0
    rep = __import__('json').load(open(out / 'report.json', encoding='utf-8'))
    seen = {r['id'] for r in rep['marks'] if r['seen_frac'] > 0.9}
    assert set(range(1, 25)) <= seen
    ft = rep['frame_times']
    assert ft['n'] == 30 and ft['n_dropped'] == 0 and ft['dt_median_ms'] == pytest.approx(1000 / FPS, abs=0.5)
    assert rep['drift']['max_px'] < 0.3
    lens = rep['lens']
    assert lens['chosen'] in ('B1', 'B2')
    b2 = lens['fits']['B2']
    assert b2['k'][0] == pytest.approx(TRUTH_K[0], abs=0.01) and b2['Zc_m'] == pytest.approx(2.90, abs=0.03)
    truth = truth_camera().corner_distortion(W, H)
    assert b2['corner_distortion'] == pytest.approx(truth, abs=0.01)
    assert lens['cross_validation_mm'][lens['chosen']]['max_mm'] < 5.0
    assert lens['homography']['max_floor_err_mm'] > lens['cross_validation_mm']['B2']['max_mm']   # distortion visible
    assert lens['checkerboard'].startswith('khong can')
    m = list(csv.DictReader(open(out / 'clip_manifest.csv', encoding='utf-8')))[0]
    assert m['sha256'] == hashlib.sha256(open(still_clip, 'rb').read()).hexdigest()
    assert (out / 'report.txt').read_text(encoding='utf-8').startswith('Chẩn đoán clip thử C7')


def test_probe_sees_a_drifting_frame(marks_csv, tmp_path):
    p = tmp_path / 'shake.mp4'
    write_clip(p, truth_camera(), n=20, shift={k: 4.0 for k in range(10, 20)})
    out = tmp_path / 'probe'
    cp.main(['--video', str(p), '--markers', marks_csv, '--out', str(out), '--stride', '1'])
    rep = __import__('json').load(open(out / 'report.json', encoding='utf-8'))
    assert rep['drift']['max_px'] > 1.5


def test_unreadable_file_reports_and_exits_2(marks_csv, tmp_path):
    bad = tmp_path / 'bad.MOV'
    bad.write_bytes(b'not a video')
    out = tmp_path / 'probe'
    assert cp.main(['--video', str(bad), '--markers', marks_csv, '--out', str(out)]) == 2
    assert 'KHÔNG mở được' in (out / 'report.txt').read_text(encoding='utf-8')


def test_refuses_a_folder_with_files(marks_csv, still_clip, tmp_path):
    (tmp_path / 'x.txt').write_text('x')
    with pytest.raises(SystemExit):
        cp.main(['--video', still_clip, '--markers', marks_csv, '--out', str(tmp_path)])
