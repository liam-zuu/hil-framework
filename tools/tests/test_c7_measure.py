"""Tests for tools/c7_measure.py on SYNTHETIC clips rendered here from a known camera (NOT a measurement).
Floor marks ID 1-24 at the print-CSV centres, check tags ID 25-33, car tag ID 0 on a plate at height h."""
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
    # mean corners kept for c7_calib_drift (CD1-3): 4 corners per fitted mark, centre = their mean
    assert set(s['ref_corners_px']) == {str(i) for i in s['fit_ids']}
    for i, c in s['ref_corners_px'].items():
        assert np.shape(c) == (4, 2)
        assert np.allclose(np.mean(c, axis=0), s['ref_centres_px'][i])


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


@pytest.mark.parametrize('off', [(0.05, 0.02, math.radians(3.0)), (-0.04, 0.03, math.radians(-5.0)),
                                 (0.0, 0.0, math.radians(4.0))])
def test_car_pose_turns_the_offset_by_the_body_heading(off):
    """Tag offset in the BODY frame (track --tag-offset, render_car above): the exact corners of a
    tag at height h, projected by the camera, give the wheel pose back. The
    old code turned (dx, dy) by the TAG heading: off by |offset| x dpsi (2.8 mm for the first case)."""
    x, y, psi = 0.6, -0.3, math.radians(25.0)
    dx, dy, dp = off
    tx, ty = x + math.cos(psi) * dx - math.sin(psi) * dy, y + math.sin(psi) * dx + math.cos(psi) * dy
    c, s = math.cos(psi + dp), math.sin(psi + dp)
    q = np.array([[tx + c * u - s * v, ty + s * u + c * v] for u, v in cm.car_corners_body(CAR_SIDE)])
    px = CAM.project(np.column_stack([q, np.full(4, TAG_H)]))
    r = cm.car_pose(CAM, px, CAR_SIDE, TAG_H / ZC, PLUMB, off)
    assert math.hypot(r['x'] - x, r['y'] - y) < 1e-6
    assert abs(cm.cp_wrap(r['psi'] - psi)) < 1e-6 and abs(cm.cp_wrap(r['tag_psi'] - psi - dp)) < 1e-6


def test_track_cli_records_the_frames_it_wrote(tmp_path, session, run_clip):
    """c7_calib_drift CD2-3: track.json carries the sha256 of the frames.csv written next to it."""
    import json
    sp = tmp_path / 'session.json'
    sp.write_text(json.dumps(cm._json(session)), encoding='utf-8')
    out = tmp_path / 'run'
    assert cm.main(['track', '--video', run_clip[0], '--session', str(sp), '--max-frames', '3',
                    '--out', str(out)]) == 0
    t = json.loads((out / 'track.json').read_text(encoding='utf-8'))
    assert t['frames_sha256'] == cm.sha256(str(out / 'frames.csv'))
    assert t['session_sha256'] == cm.sha256(str(sp)) and t['frames'] == 3


def _gate_rows(n=60, scale=lambda k: 0.976):
    return [dict(t_s=k / 60, n_marks=20, rms_px=0.3, car_seen=True, res_m=0.0025, res_sim_m=0.0003,
                 scale=scale(k), x=0.01 * k, y=0.0, psi=0.0) for k in range(n)]


def test_gates_shape_scale_and_no_position_gate():
    """Liam 16/11 CODE_CAM (Q1, Q2): (2a) on the residual WITH scale, (2b) against the neighbours' median; a
    uniformly shrunken tag (rigid residual 2.5 mm, the 16/11 trial) passes; a jump of the position is no gate
    (O1 v11.10 RS2-1, c7_score marks it)."""
    rows = _gate_rows(scale=lambda k: 0.976 - 0.008 * k / 60)        # slow drift (blur with speed): 0.8 % in 1 s
    rows[5]['n_marks'] = 6
    rows[7]['res_sim_m'] = 0.004
    rows[30]['scale'] *= 1.01                                         # a sudden 1 % jump: a misread
    rows[40]['x'] += 0.05                                             # 50 mm jump: kept (c7_score)
    cm.apply_gates(rows, rms_h=0.3)
    bad = {k: r['why'] for k, r in enumerate(rows) if not r['valid']}
    assert bad == {5: ['moc_it'], 7: ['xe_meo'], 30: ['ti_le']}
    assert all(np.isfinite(r['speed']) for r in rows if r['valid'])
    assert abs(rows[20]['scale_ref'] - rows[20]['scale']) < 0.001


def test_old_gates_would_drop_the_shrunk_tag_and_the_drift():
    """Mutation guard: the residual WITHOUT scale (2.5 mm) or a clip-wide median would fail these frames."""
    rows = _gate_rows(scale=lambda k: 0.976 - 0.012 * k / 60)
    cm.apply_gates(rows, rms_h=0.3)
    assert all(r['valid'] for r in rows)
    sc = np.array([r['scale'] for r in rows])
    assert np.max(np.abs(sc / np.median(sc) - 1.0)) > cm.GATE_SCALE_REL           # a clip median would fail some
    assert all(r['res_m'] > cm.GATE_CAR_RES_M for r in rows)                       # so would the rigid residual


def test_scale_gate_needs_neighbours():
    rows = _gate_rows(n=8)
    cm.apply_gates(rows, rms_h=0.3)
    assert all(r['why'] == ['ti_le_it_lan_can'] for r in rows)
    rows = _gate_rows(n=9)
    cm.apply_gates(rows, rms_h=0.3)
    assert all(r['valid'] for r in rows)


def test_scale_window_is_time_not_frames():
    """O1 v11.30 3.3 (2): neighbours within +-0.25 s. A frame alone after a 1 s hole of frames failing (2a) has 15
    passing frames before it by index but none within 0.25 s: it fails (an index window would pass it)."""
    rows = _gate_rows(n=120)
    for k in range(19, 90):
        rows[k]['res_sim_m'] = 0.01
    rows[90]['res_sim_m'] = 0.0003
    for k in range(91, 120):
        rows[k]['res_sim_m'] = 0.01
    cm.apply_gates(rows, rms_h=0.3)
    assert rows[90]['why'] == ['ti_le_it_lan_can'] and rows[10]['valid']


def test_scale_window_at_120_fps():
    """120 fps: +-0.25 s holds +-30 frames; a 1 % jump is still caught, a slow drift is not."""
    rows = [dict(r, t_s=k / 120) for k, r in enumerate(_gate_rows(n=240, scale=lambda k: 0.976 - 0.008 * k / 120))]
    rows[100]['scale'] *= 1.01
    cm.apply_gates(rows, rms_h=0.3)
    assert {k for k, r in enumerate(rows) if not r['valid']} == {100}
    nb = sum(1 for r in rows if abs(r['t_s'] - rows[50]['t_s']) <= 0.25 + 1e-9) - 1
    assert nb == 60


def test_scale_gate_ignores_frames_failing_shape_or_floor():
    rows = _gate_rows()
    for k in range(20, 40):
        rows[k]['res_sim_m'] = 0.01                                    # 20 frames fail (2a): not neighbours
    rows[19]['scale'] *= 1.01
    cm.apply_gates(rows, rms_h=0.3)
    assert rows[19]['why'] == ['ti_le'] and rows[40]['valid']


@pytest.mark.parametrize('shrink', [0.0, 0.024, -0.01])
def test_res_sim_of_a_uniformly_shrunk_tag(shrink):
    """The corners of a tag drawn (1 - shrink) times its size: the fit WITH scale has no residual, the rigid one
    has (1 - scale) side / sqrt 2 per corner, and res_sim_from gives the residual with scale back from it."""
    x, y, psi = 0.4, 0.2, math.radians(30.0)
    c, s = math.cos(psi), math.sin(psi)
    q = np.array([[x + (1 - shrink) * (c * u - s * v), y + (1 - shrink) * (s * u + c * v)]
                  for u, v in cm.car_corners_body(CAR_SIDE)])
    px = CAM.project(np.column_stack([q, np.full(4, TAG_H)]))
    r = cm.car_pose(CAM, px, CAR_SIDE, TAG_H / ZC, PLUMB)
    assert r['res_sim_m'] < 1e-6 and abs(r['scale'] - (1 - shrink)) < 1e-6
    assert abs(r['res_m'] - abs(shrink) * CAR_SIDE / math.sqrt(2.0)) < 1e-6
    assert abs(cm.res_sim_from(r['res_m'], r['scale'], CAR_SIDE) - r['res_sim_m']) < 1e-6
    assert math.hypot(r['x'] - x, r['y'] - y) < 1e-6 and abs(cm.cp_wrap(r['psi'] - psi)) < 1e-6


def test_res_sim_from_a_distorted_tag():
    """One corner moved 2 mm: res_sim_from (from res_m, scale) equals the residual of the fit with scale."""
    x, y, psi = -0.3, 0.5, math.radians(-40.0)
    c, s = math.cos(psi), math.sin(psi)
    q = np.array([[x + 0.98 * (c * u - s * v), y + 0.98 * (s * u + c * v)] for u, v in cm.car_corners_body(CAR_SIDE)])
    q[1] += [0.002, -0.001]
    px = CAM.project(np.column_stack([q, np.full(4, TAG_H)]))
    r = cm.car_pose(CAM, px, CAR_SIDE, TAG_H / ZC, PLUMB)
    assert r['res_sim_m'] > 5e-4
    assert abs(cm.res_sim_from(r['res_m'], r['scale'], CAR_SIDE) - r['res_sim_m']) < 1e-9


def test_regate_cli_on_an_old_frames_file(tmp_path):
    """regate: a frames.csv of the old version (pred_m, no res_sim_m, scale_ref) gets res_sim_m and the new gates."""
    import json
    side = CAR_SIDE
    hdr = ('idx,t_s,valid,why,n_marks,rms_px,shift_px,cam_z,car_seen,x,y,psi,tag_x,tag_y,res_m,scale,pred_m,row_px,'
           'vx,vy,wz,speed')
    lines = [hdr]
    for k in range(40):
        sc = 0.976
        res = (1 - sc) * side / math.sqrt(2.0)                        # pure uniform shrink: res_sim 0
        lines.append(f'{k},{k / 60:.6f},0,xe_du_thua,24,0.3,1.0,2.6,1,{0.01 * k:.6f},0,0,{0.01 * k:.6f},0,'
                     f'{res:.6f},{sc:.6f},,500,,,,')
    (tmp_path / 'old.csv').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (tmp_path / 's.json').write_text(json.dumps(dict(car_side_m=side, rms_px_H=0.3)), encoding='utf-8')
    assert cm.main(['regate', '--frames', str(tmp_path / 'old.csv'), '--session', str(tmp_path / 's.json'),
                    '--out', str(tmp_path / 'new')]) == 0
    rows = cm.read_frames(str(tmp_path / 'new' / 'frames.csv'))
    # res_m, scale have 6 decimals in the CSV: near 0, sqrt(res^2 - x^2) magnifies the rounding to about
    # sqrt(2 res 5e-7) = 0.05 mm (regate docstring); far below the 1.5 mm gate
    assert all(r['valid'] for r in rows) and all(r['res_sim_m'] < 1e-4 for r in rows)
    t = json.loads((tmp_path / 'new' / 'track.json').read_text(encoding='utf-8'))
    assert t['n_valid'] == 40 and t['regate_of_sha256'] == cm.sha256(str(tmp_path / 'old.csv'))


# ---------------------------------------------------------------- Codex vong 1 (16/11): CS1-13, CS1-15, CS1-17
def _cs1_rows(n=20, res_m=0.0002, res_sim_m=0.0002, scale=1.0):
    return [dict(idx=k, t_s=k / 60.0, n_marks=24, rms_px=0.3, car_seen=True, res_m=res_m, res_sim_m=res_sim_m,
                 scale=scale, x=0.001 * k, y=0.0, psi=0.0) for k in range(n)]


def test_cs1_13_shift_is_the_length_of_the_median_vector():
    """O1 3.5: 'do lon cua trung vi vec-to dich tam': 8 shifts of 13 px in four directions have a median vector 0
    (the old median of the lengths gave 13 px and a bumped camera)."""
    v = [(13, 0), (-13, 0), (0, 13), (0, -13)] * 2
    sh, mag = cm.shift_of(v)
    assert sh == pytest.approx(0.0) and mag == pytest.approx(13.0)
    sh, mag = cm.shift_of([(3, 4)] * 5)
    assert sh == pytest.approx(5.0) and mag == pytest.approx(5.0)


def test_cs1_15_gates_from_the_constants():
    rows = _cs1_rows(res_sim_m=0.0012)
    cm.apply_gates(rows, 0.3)
    assert all(r['valid'] for r in rows)
    g = dict(cm.DEFAULT_GATES, car_res_sim_m=0.001)
    cm.apply_gates(rows, 0.3, g)
    assert not any(r['valid'] for r in rows) and all('xe_meo' in r['why'] for r in rows)
    with pytest.raises(ValueError):
        cm.check_gates(dict(cm.DEFAULT_GATES, scale_min=2.5))


def test_cs1_17_regate_rounding_that_can_flip_the_shape_gate():
    """Codex CS1-17: side 0.1444 m, scale 0.976, true residual with scale 1.5001 mm -> rigid 2.873237145 mm, written
    0.002873 in an old frames.csv: the value rebuilt from it (1.4996 mm) passes, the true one fails; the rounding
    interval straddles 1.5 mm, so regate cannot decide: 'can_track_lai'."""
    side = 0.1444
    rig = math.sqrt(0.0015001 ** 2 + ((1 - 0.976) * side) ** 2 / 2)
    assert rig == pytest.approx(0.002873237145, abs=1e-12)
    rows = _cs1_rows(res_m=round(rig, 6), res_sim_m=float('nan'), scale=0.976)
    cm.regate(rows, dict(car_side_m=side, rms_px_H=0.3))
    assert all(r['why'] == ['can_track_lai'] for r in rows) and all(r['res_sim_m'] < 0.0015 for r in rows)
    rows = _cs1_rows(res_m=0.0005, res_sim_m=float('nan'), scale=0.999)        # far from the threshold
    cm.regate(rows, dict(car_side_m=side, rms_px_H=0.3))
    assert all(r['valid'] for r in rows)


def test_cs1_17_regate_scale_ratio_on_the_threshold():
    rows = _cs1_rows(scale=1.0)
    rows[10]['scale'] = 1.005                            # exactly 0.5 % from the median: rounding decides
    cm.regate(rows, dict(car_side_m=0.1444, rms_px_H=0.3))
    assert rows[10]['why'] == ['can_track_lai'] and rows[9]['valid']
    rows = _cs1_rows(scale=1.0)
    rows[10]['scale'] = 1.006
    cm.regate(rows, dict(car_side_m=0.1444, rms_px_H=0.3))
    assert rows[10]['why'] == ['ti_le']


def test_cs1_15_17_regate_cli_with_constants(tmp_path):
    rows = _cs1_rows()
    for r in rows:
        r.update(why=[], valid=True, shift_px=0.2, cam_z=2.6, tag_x=0.0, tag_y=0.0, scale_ref=1.0, row_px=1.0,
                 vx=0.0, vy=0.0, wz=0.0, speed=0.0)
    cm.write_frames(tmp_path / 'frames.csv', rows)
    head = (tmp_path / 'frames.csv').read_text(encoding='utf-8').splitlines()
    assert head[1].split(',')[cm.FRAME_COLS.index('res_m')] == '0.0002000000'            # 10 decimals now
    (tmp_path / 'session.json').write_text(json.dumps(dict(car_side_m=0.1444, rms_px_H=0.3)), encoding='utf-8')
    hs = dict(version='c7_hang_so_v1', trang_thai='de_xuat', cong_khung=dict(cm.DEFAULT_GATES, car_res_sim_m=0.0001))
    (tmp_path / 'hs.json').write_text(json.dumps(hs), encoding='utf-8')
    assert cm.main(['regate', '--frames', str(tmp_path / 'frames.csv'), '--session', str(tmp_path / 'session.json'),
                    '--hang-so', str(tmp_path / 'hs.json'), '--out', str(tmp_path / 'out')]) == 0
    info = json.loads((tmp_path / 'out' / 'track.json').read_text(encoding='utf-8'))
    assert info['gates']['car_res_sim_m'] == 0.0001 and info['gates']['hang_so']['file'] == 'hs.json'
    assert info['n_valid'] == 0 and info['shift_kind'] == 'median_magnitude'      # no shift_mag_px column written
