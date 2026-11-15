"""c7_calib_drift.py: compare the clips H of a session, score twice (O1 v11.27 section 4; design
CLAUDE_THESIS 03_CODE/docs/2026-11-15_Thiet_ke_so_clip_H_dau_cuoi_v1.md). Sessions are synthetic: a
truth camera sees the TRUE mark positions, the session is fitted to the mark FILE (as calibrate does)."""
import json
import math
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import c7_calib_drift as cd  # noqa: E402
import c7_clip_probe as cp  # noqa: E402
import c7_measure as cm  # noqa: E402
from test_c7_clip_probe import MARKS, SIDES, W, H, truth_camera  # noqa: E402


def file_marks():
    m = {}
    for group, side in zip(MARKS, SIDES):
        for i, x, y in group:
            m[i] = (x / 100, y / 100, side / 1000)
    return m


def make_session(cam_true, true_marks=None, video='H.mov', zc=2.90, drop=()):
    """Session dict as c7_measure.calibrate writes it: camera fitted to the mark FILE, reference
    centres = mean of the corners the truth camera sees at the TRUE positions."""
    fm = file_marks()
    tm = dict(fm) if true_marks is None else true_marks
    ids = [i for i in sorted(fm) if i not in drop]
    img = np.vstack([cam_true.project(np.column_stack([cp.mark_corners_floor(*tm[i]), np.zeros(4)])) for i in ids])
    obj = np.vstack([cp.mark_corners_floor(*fm[i]) for i in ids])
    cam, rms = cm.fit_two_pass(cam_true.model, obj, img, W, H, zc)
    ref = {str(i): img[4 * k:4 * k + 4].mean(axis=0).tolist() for k, i in enumerate(ids)}
    return dict(kind='c7_session', video=video, video_sha256='x' * 64, camera=cm.cam_to_dict(cam), rms_px_H=rms,
                ref_centres_px=ref, marks={str(i): list(v) for i, v in fm.items()}, zc_fit_m=float(abs(cam.centre[2])))


def moved_camera(dx=0.02, dyaw_deg=0.5):
    """Truth camera after a rigid move: centre shifted, a small extra yaw (no intrinsics change)."""
    c = truth_camera()
    C = c.centre + np.array([dx, -dx / 2, 0.01])
    R = cv2.Rodrigues(np.array([0.0, 0.0, math.radians(dyaw_deg)]))[0] @ c.R
    return cp.Camera(c.model, c.f, c.cx, c.cy, c.k, cv2.Rodrigues(R)[0].ravel(), -R @ C)


@pytest.fixture(scope='module')
def s0():
    return make_session(truth_camera())


def test_lane_points_cover_figure_8_and_k_lanes():
    p8, n8 = cd.lane_points()
    assert n8 == ['O1-F'] and len(p8) > 50
    assert np.abs(p8[:, 0]).max() == pytest.approx(1.0, abs=0.06) and np.abs(p8[:, 1]).max() > 0.4
    pk, nk = cd.lane_points((0.2, -0.1, math.radians(30)))
    assert nk == ['O1-F', 'O1-K1', 'O1-K2'] and len(pk) > len(p8)


def test_same_session_gives_zero(s0):
    d = cd.compare(s0, json.loads(json.dumps(s0)))
    assert d['D_H_mm'] < 0.01 and not d['vuot'] and d['marks_grown'] == []
    assert d['shift_px']['max'] == pytest.approx(0.0, abs=1e-9)


def test_rigid_camera_move_is_compensated(s0):
    """track refits the pose per frame (design 3.6): a rigid move alone must not trip the threshold."""
    s1 = make_session(moved_camera())
    d = cd.compare(s0, s1)
    assert d['shift_px']['median'] > 5.0                     # the marks moved in the image
    assert d['D_H_mm'] < 0.2 and not d['vuot']
    assert d['marks_grown'] == []


def test_focal_length_change_is_seen(s0):
    c = truth_camera()
    zoom = cp.Camera(c.model, c.f * 1.02, c.cx, c.cy, c.k, c.rvec, c.tvec)
    calm = cd.compare(s0, make_session(truth_camera()))
    d = cd.compare(s0, make_session(zoom))
    assert d['focal_rel'] == pytest.approx(0.02, abs=0.005)
    assert d['D_H_mm'] > 10 * max(calm['D_H_mm'], 0.01)


def test_distortion_change_trips_the_threshold(s0):
    c = truth_camera()
    warp = cp.Camera(c.model, c.f, c.cx, c.cy, (c.k[0] + 0.03, c.k[1]), c.rvec, c.tvec)
    d = cd.compare(s0, make_session(warp))
    assert d['vuot'] and d['D_H_mm'] > cd.TH_MM


def test_d_h_is_the_larger_of_both_directions(s0):
    c = truth_camera()
    d = cd.compare(s0, make_session(cp.Camera(c.model, c.f, c.cx, c.cy, (c.k[0] - 0.03, c.k[1]), c.rvec, c.tvec)))
    a, b = d['D_mm']['h0_to_h1']['max'], d['D_mm']['h1_to_h0']['max']
    assert b > a + 0.02                                     # this case is not symmetric
    assert d['D_H_mm'] == pytest.approx(max(a, b))


def test_moved_floor_mark_is_named(s0):
    tm = file_marks()
    x, y, side = tm[14]
    tm[14] = (x + 0.015, y - 0.010, side)                   # mark 14 kicked by 18 mm
    d = cd.compare(s0, make_session(truth_camera(), true_marks=tm))
    assert [m['id'] for m in d['marks_grown']] == [14]
    assert d['shift_px']['max_id'] == 14


def test_missing_mark_in_h1_is_listed(s0):
    d = cd.compare(s0, make_session(truth_camera(), drop=(24,)))
    assert d['missing_in_h1'] == [24] and d['D_H_mm'] < 0.5


def test_threshold_uses_the_repeat_noise(s0):
    d = cd.compare(s0, json.loads(json.dumps(s0)), d_lap_mm=1.6)
    assert d['T_H_mm'] == pytest.approx(3.2)
    d = cd.compare(s0, json.loads(json.dumps(s0)), d_lap_mm=0.4)
    assert d['T_H_mm'] == pytest.approx(cd.TH_MM)


def test_missing_h1(s0):
    d = cd.compare(s0, None)
    assert d['thieu_H1'] and d['vuot'] and d['D_H_mm'] is None


def test_different_mark_files_refused(s0):
    s1 = json.loads(json.dumps(s0))
    s1['marks']['3'][0] += 0.001
    with pytest.raises(ValueError):
        cd.compare(s0, s1)


# ---------------------------------------------------------------- combine
def sc(outcome, t0=1000, off=0.25, e=0.01):
    return dict(outcome=outcome, outcome_reasons=[], conditions={}, e_perp_max=e, completed='true',
                t0_log_us=t0, offset_s=off, frames_valid=600, gap_max=0.02, shift_max_px=0.5)


DRIFT_OK = dict(kind='c7_calib_drift', vuot=False, thieu_H1=False, D_H_mm=0.5, T_H_mm=2.0)
DRIFT_OVER = dict(kind='c7_calib_drift', vuot=True, thieu_H1=False, D_H_mm=3.1, T_H_mm=2.0)
DRIFT_MISS = dict(kind='c7_calib_drift', vuot=True, thieu_H1=True, D_H_mm=None, T_H_mm=2.0)


@pytest.mark.parametrize('o0,o1,final,split', [
    ('dat', 'dat', 'dat', False), ('khong_dat', 'khong_dat', 'khong_dat', False),
    ('chua_ket_luan', 'chua_ket_luan', 'chua_ket_luan', False),
    ('dat', 'khong_dat', 'chua_ket_luan', True), ('khong_dat', 'dat', 'chua_ket_luan', True),
    ('chua_ket_luan', 'dat', 'chua_ket_luan', True), ('dat', 'chua_ket_luan', 'chua_ket_luan', True),
    ('khong_dat', 'chua_ket_luan', 'chua_ket_luan', True),
    ('hong_he_do', 'dat', 'hong_he_do', False), ('dat', 'hong_he_do', 'hong_he_do', False),
    ('hong_he_do', 'hong_he_do', 'hong_he_do', False)])
def test_combine_table(o0, o1, final, split):
    r = cd.combine(DRIFT_OVER, sc(o0), sc(o1))
    assert r['outcome'] == final
    assert 'calib_drift' in r['calib_drift']['flags']
    assert ('calib_drift_split' in r['calib_drift']['flags']) == split
    assert r['calib_drift']['outcome_h0'] == o0 and r['calib_drift']['outcome_h1'] == o1


def test_combine_below_threshold_keeps_h0():
    r = cd.combine(DRIFT_OK, sc('dat'))
    assert r['outcome'] == 'dat' and r['calib_drift']['flags'] == []
    with pytest.raises(ValueError):
        cd.combine(DRIFT_OK, sc('dat'), sc('khong_dat'))


@pytest.mark.parametrize('o0,final', [('dat', 'chua_ket_luan'), ('khong_dat', 'chua_ket_luan'),
                                      ('chua_ket_luan', 'chua_ket_luan'), ('hong_he_do', 'hong_he_do')])
def test_combine_missing_h1(o0, final):
    r = cd.combine(DRIFT_MISS, sc(o0))
    assert r['outcome'] == final and r['calib_drift']['flags'] == ['thieu_H1']
    with pytest.raises(ValueError):
        cd.combine(DRIFT_MISS, sc(o0), sc(o0))


def test_combine_checks_same_run_and_clock():
    with pytest.raises(ValueError):
        cd.combine(DRIFT_OVER, sc('dat'))                                  # second score missing
    with pytest.raises(ValueError):
        cd.combine(DRIFT_OVER, sc('dat', t0=1000), sc('dat', t0=2000))     # another log
    with pytest.raises(ValueError):
        cd.combine(DRIFT_OVER, sc('dat', off=0.25), sc('dat', off=0.26))   # another clock offset
    with pytest.raises(ValueError):
        cd.combine(DRIFT_OVER, sc('dat'), sc('xyz'))


def test_cli_round_trip(tmp_path, s0, capsys):
    p0, p1 = tmp_path / 's0.json', tmp_path / 's1.json'
    p0.write_text(json.dumps(s0), encoding='utf-8')
    c = truth_camera()
    p1.write_text(json.dumps(make_session(cp.Camera(c.model, c.f, c.cx, c.cy, (c.k[0] + 0.03, c.k[1]),
                                                    c.rvec, c.tvec))), encoding='utf-8')
    dr = tmp_path / 'drift.json'
    assert cd.main(['so', '--s0', str(p0), '--s1', str(p1), '--out', str(dr)]) == 0
    assert capsys.readouterr().out.rstrip().endswith('#ket_qua,ma=0,vuot=1')
    assert cd.main(['so', '--s0', str(p0), '--s1', str(p1), '--out', str(dr)]) == 2    # no overwrite
    capsys.readouterr()
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    a.write_text(json.dumps(sc('dat')), encoding='utf-8')
    b.write_text(json.dumps(sc('khong_dat')), encoding='utf-8')
    out = tmp_path / 'final.json'
    assert cd.main(['gop', '--drift', str(dr), '--s0-score', str(a), '--s1-score', str(b), '--out', str(out)]) == 0
    assert capsys.readouterr().out.rstrip().endswith('#ket_qua,ma=0,outcome=chua_ket_luan')
    r = json.loads(out.read_text(encoding='utf-8'))
    assert r['outcome'] == 'chua_ket_luan' and r['calib_drift']['h0']['outcome'] == 'dat'
    miss = tmp_path / 'miss.json'
    assert cd.main(['so', '--s0', str(p0), '--out', str(miss)]) == 0
    assert capsys.readouterr().out.rstrip().endswith('#ket_qua,ma=0,vuot=1')
