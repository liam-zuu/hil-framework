"""c7_calib_drift.py: compare the clips H of a session, score twice (O1 v11.28 section 4; design
CLAUDE_THESIS 03_CODE/docs/2026-11-15_Thiet_ke_so_clip_H_dau_cuoi_v1.md; Codex Sol round 1 CD1-2..11,
round 2 CD2-1..7, round 3 CD3-2..6). Sessions are synthetic: a truth camera sees the TRUE mark positions, the session is
fitted to the mark FILE (as calibrate does)."""
import itertools
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
import c7_score as cs  # noqa: E402
import o1_lanes  # noqa: E402
from o1_feasibility import HEAD0  # noqa: E402
from test_c7_clip_probe import MARKS, SIDES, W, H, truth_camera  # noqa: E402

TAG_H, ZC_TAPE, PLUMB, SIDE = 0.12, 2.90, [0.03, -0.02], 0.1445
FIG = ['O1-F02', 'O1-F05', 'O1-F08']


def file_marks():
    m = {}
    for group, side in zip(MARKS, SIDES):
        for i, x, y in group:
            m[i] = (x / 100, y / 100, side / 1000)
    return m


def make_session(cam_true, true_marks=None, video='H.mov', drop=(), sha='s0', **over):
    """Session dict as c7_measure.calibrate writes it: camera fitted to the mark FILE, reference corners
    = what the truth camera sees at the TRUE positions."""
    fm = file_marks()
    tm = dict(fm) if true_marks is None else true_marks
    ids = [i for i in sorted(fm) if i not in drop]
    img = np.vstack([cam_true.project(np.column_stack([cp.mark_corners_floor(*tm[i]), np.zeros(4)])) for i in ids])
    obj = np.vstack([cp.mark_corners_floor(*fm[i]) for i in ids])
    cam, rms = cm.fit_two_pass(cam_true.model, obj, img, W, H, ZC_TAPE)
    s = dict(kind='c7_session', video=video, video_sha256='v' * 64, camera=cm.cam_to_dict(cam), rms_px_H=rms,
             ref_centres_px={str(i): img[4 * k:4 * k + 4].mean(axis=0).tolist() for k, i in enumerate(ids)},
             ref_corners_px={str(i): img[4 * k:4 * k + 4].tolist() for k, i in enumerate(ids)},
             marks={str(i): list(v) for i, v in fm.items()}, zc_fit_m=float(abs(cam.centre[2])),
             zc_tape_m=ZC_TAPE, plumb_xy_m=list(PLUMB), tag_h_m=TAG_H, car_side_m=SIDE, width=W, height=H,
             kappa=TAG_H / ZC_TAPE, markers_csv=[dict(file='moc_san_do.csv', sha256='m' * 64)], _sha256=sha,
             tag_offset=[0.0, 0.0, 0.0])
    s.update(over)
    return s


def moved_camera(d=(0.02, -0.01, 0.0), dyaw_deg=0.5):
    """Truth camera after a rigid move: centre shifted, a small extra yaw (no intrinsics change)."""
    c = truth_camera()
    C = c.centre + np.array(d)
    R = cv2.Rodrigues(np.array([0.0, 0.0, math.radians(dyaw_deg)]))[0] @ c.R
    return cp.Camera(c.model, c.f, c.cx, c.cy, c.k, cv2.Rodrigues(R)[0].ravel(), -R @ C)


def warped(dk1):
    c = truth_camera()
    return cp.Camera(c.model, c.f, c.cx, c.cy, (c.k[0] + dk1, c.k[1]), c.rvec, c.tvec)


def zoomed(rel):
    c = truth_camera()
    return cp.Camera(c.model, c.f * (1 + rel), c.cx, c.cy, c.k, c.rvec, c.tvec)


@pytest.fixture(scope='module')
def s0():
    return make_session(truth_camera())


# ---------------------------------------------------------------- lanes (CD1-8, CD1-9, CD2-7)
def test_lane_points_need_the_tasks_and_k_start():
    p8, n8 = cd.lane_points(FIG)
    assert n8 == ['O1-F'] and len(p8) > 50
    assert np.abs(p8[:, 0]).max() == pytest.approx(1.0, abs=0.06) and np.abs(p8[:, 1]).max() > 0.4
    with pytest.raises(ValueError):
        cd.lane_points(['O1-F05', 'O1-K1'])                 # K1 without its start pose
    pk, nk = cd.lane_points(['O1-F05', 'O1-K1', 'O1-K2'], (0.2, -0.1, math.radians(30)))
    assert nk == ['O1-F', 'O1-K1', 'O1-K2'] and len(pk) > len(p8)
    only_k, nok = cd.lane_points(['O1-K2'], (0.0, 0.0, 0.0))
    assert nok == ['O1-K2']
    with pytest.raises(ValueError):
        cd.parse_tasks('O1-F05,O1-X9')
    with pytest.raises(ValueError):
        cd.parse_tasks(' , ')


@pytest.mark.parametrize('task', ['O1-K1', 'O1-K2'])
def test_k_lane_is_placed_on_the_start_pose(task):
    """CD2-7: the placed K points, taken back to the lane frame by c7_score.KTask.to_lane (the scorer's
    own inverse), are the lane points every 5 cm; and they are not the unplaced lane."""
    start = (0.7, -0.4, math.radians(-120))
    p, _ = cd.lane_points([task], start)
    kt = cs.KTask(task, start)
    back = np.array([kt.to_lane(x, y, 0.0)[:2] for x, y in p])
    lane = o1_lanes.Lane(o1_lanes.k_corners(task))
    want = cd._resample(lane.s, lane.x, lane.y, cd.STEP_M)
    assert back.shape == want.shape and np.allclose(back, want, atol=1e-9)
    assert np.allclose(p[0], start[:2]) and np.abs(p - want).max() > 0.3


@pytest.mark.parametrize('task', ['O1-K1', 'O1-K2'])
def test_k_body_heading_is_held(task):
    """CD3-3: K1 strafes, K2 drives a square without turning (o1_tasks): the body keeps the heading of
    the nominal start pose everywhere, corners of K2 included."""
    h0 = math.radians(40)
    p, _ = cd.lane_poses([task], (0.3, 0.1, h0))
    assert np.allclose(p[:, 2], h0)


def test_figure_8_body_heading_is_the_tangent():
    """o1_tasks: heading on the tangent; start at HEAD0 (45 deg, c7_score.Fig8Task.start); at x = A the
    path goes straight down (dx/dtau = 0, dy/dtau = -A)."""
    p, _ = cd.lane_poses(['O1-F05'])
    assert cm.cp_wrap(p[0, 2] - HEAD0) == pytest.approx(0.0, abs=1e-3)
    k = int(np.argmax(p[:, 0]))
    assert cm.cp_wrap(p[k, 2] + math.pi / 2) == pytest.approx(0.0, abs=0.05)
    d = np.diff(p[:, :2], axis=0)                                # independent: chord directions
    chord = np.arctan2(d[:, 1], d[:, 0])
    mid = (p[1:, 2] + p[:-1, 2]) / 2
    assert np.max(np.abs([cm.cp_wrap(a - b) for a, b in zip(chord, mid)])) < 0.1


def test_lane_samples_every_5_cm_and_keep_the_end():
    s = np.linspace(0.0, 1.03, 1031)
    q = cd._resample(s, s, np.zeros_like(s), 0.05)
    d = np.diff(q[:, 0])
    assert np.allclose(d[:-1], 0.05) and d[-1] == pytest.approx(0.03) and q[-1, 0] == pytest.approx(1.03)


# ---------------------------------------------------------------- compare
def truth_poses(tasks, ks):
    """Wheel poses built here, not with lane_poses (CD3-3): the lane points (checked against
    c7_score.KTask.to_lane above), body heading = chord direction on the figure 8, k_start heading on
    K1, K2 (o1_tasks)."""
    out = []
    if any(t in cd.FIG8 for t in tasks):
        q, _ = cd.lane_points(['O1-F05'])
        d = np.gradient(q, axis=0)
        out.append(np.column_stack([q, np.arctan2(d[:, 1], d[:, 0])]))
    for t in tasks:
        if t in cd.FINAL_SET:
            q, _ = cd.lane_points([t], ks)
            out.append(np.column_stack([q, np.full(len(q), ks[2])]))
    return np.vstack(out)


def truth_error_mm(s_score, cam_true, poses, off=(0.0, 0.0, 0.0)):
    """Independent of one_way (CD2-7 control for CD2-1, CD2-2, CD3-2): the TRUTH camera of the end of
    the session images the car tag corners at their height (tag centre = wheel + R(psi) (dx, dy), tag
    heading psi + dpsi) and the floor marks at their true place; the scoring session's camera is
    refitted to those mark corners and c7_measure.car_pose runs with the session's kappa and plumb
    point and the tag offset. Largest wheel centre error [mm]."""
    fm = file_marks()
    ids = sorted(fm)
    obj = np.vstack([cp.mark_corners_floor(*fm[i]) for i in ids])
    img = cam_true.project(np.column_stack([obj, np.zeros(len(obj))]))
    a2, _ = cm.refit_pose(cm.cam_from_dict(s_score['camera']), obj, img)
    worst = 0.0
    dx, dy, dp = off
    for x, y, psi in poses:
        tx, ty = x + math.cos(psi) * dx - math.sin(psi) * dy, y + math.sin(psi) * dx + math.cos(psi) * dy
        c, s = math.cos(psi + dp), math.sin(psi + dp)
        q = np.array([[tx + c * u - s * v, ty + s * u + c * v] for u, v in cm.car_corners_body(SIDE)])
        px = cam_true.project(np.column_stack([q, np.full(4, TAG_H)]))
        r = cm.car_pose(a2, px, SIDE, s_score['kappa'], s_score['plumb_xy_m'], off)
        worst = max(worst, math.hypot(r['x'] - x, r['y'] - y))
    return 1000.0 * worst


def test_same_session_gives_zero(s0):
    d = cd.compare(s0, json.loads(json.dumps(s0)), FIG)
    assert d['D_H_mm'] < 0.01 and not d['vuot'] and d['marks_grown'] == []
    assert d['shift_px']['max'] == pytest.approx(0.0, abs=1e-9)


def test_rigid_camera_move_leaves_only_the_parallax_term(s0):
    """track refits the pose per frame; the plumb point of the session does not move (CD1-2):
    a 22 mm lateral move leaves about kappa x 22 mm on the car."""
    s1 = make_session(moved_camera())
    d = cd.compare(s0, s1, FIG)
    assert d['shift_px']['median'] > 5.0
    expect = 1000 * (TAG_H / ZC_TAPE) * math.hypot(0.02, 0.01)
    assert d['D_H_mm'] == pytest.approx(expect, rel=0.1)
    assert not d['vuot'] and d['marks_grown'] == []
    assert np.hypot(*d['camera_shift_m'][:2]) == pytest.approx(math.hypot(0.02, 0.01), rel=0.1)


K_FAR = (2.7, -1.0, math.radians(90))                      # K1 along x = 2.7 m (Codex CD2-1)
K_NEAR = (1.0, 0.0, math.radians(-90))                     # Codex CD2-2 (k1 -0.050 -> -0.049)


@pytest.mark.parametrize('name,cam,tasks,ks', [
    ('lateral', moved_camera(), FIG, None),
    ('rise_6cm', moved_camera((0.0, 0.0, 0.06), 0.0), FIG + ['O1-K1'], K_FAR),
    ('k1_0.001', warped(0.001), FIG + ['O1-K1'], K_NEAR),
    ('k1_0.01', warped(0.01), FIG, None),
    ('zoom_2pc', zoomed(0.02), FIG, None)])
def test_d_h_covers_the_car_pose_error_of_the_truth_camera(s0, name, cam, tasks, ks):
    """CD2-1, CD2-2: D_H (H1 frames scored with H0) is never below the error car_pose makes with the
    truth camera; where the fit sees the change (no height ambiguity) it is that error."""
    d = cd.compare(s0, make_session(cam), tasks, ks)
    truth = truth_error_mm(s0, cam, truth_poses(tasks, ks))
    way = d['D_mm']['h0_to_h1']
    assert max(v['max'] for v in way.values()) >= truth - 0.05
    assert d['D_H_mm'] >= truth - 0.05
    if name != 'rise_6cm':
        assert way['fit']['max'] == pytest.approx(truth, abs=0.05)
    assert d['vuot'] == (d['D_H_mm'] > cd.EPS_MM)


def test_height_change_held_by_the_tape_is_caught(s0):
    """CD2-1: both fits hold Z_c to the tape, so a 6 cm rise shows as a focal change; read as a height
    change it gives about the true error (above eps_H on the far K lane), the fit alone does not."""
    cam = moved_camera((0.0, 0.0, 0.06), 0.0)
    d = cd.compare(s0, make_session(cam), FIG + ['O1-K1'], K_FAR)
    truth = truth_error_mm(s0, cam, truth_poses(FIG + ['O1-K1'], K_FAR))
    assert truth > cd.EPS_MM
    assert d['D_mm']['h0_to_h1']['fit']['max'] < 0.5 * truth            # the fit alone misses it
    assert d['D_mm']['h0_to_h1']['focal_as_height']['max'] == pytest.approx(truth, rel=0.1)
    assert d['lens_z_m']['h0_to_h1']['focal_as_height'] == pytest.approx(ZC_TAPE + 0.06, abs=0.01)
    assert d['vuot'] and d['worst']['hypothesis'] == 'focal_as_height'


@pytest.mark.parametrize('off', [(0.03, 0.0, 0.0), (0.03, 0.01, math.radians(2.0))])
def test_tag_offset_enters_the_drift(s0, off):
    """CD3-2 (Codex: K1 at 2.7,-1,0, k1 -0.050 -> -0.0491, offset (0.03, 0, 0)): the drift is computed
    with the tag offset the runs are tracked with, and covers the truth error with that offset (body
    frame offset and a tag heading offset, as c7_measure.car_pose takes it)."""
    ks = (2.7, -1.0, 0.0)
    cam = warped(0.0009)
    s1 = make_session(cam)
    d0 = cd.compare(s0, s1, ['O1-K1'], ks)
    d = cd.compare(s0, s1, ['O1-K1'], ks, tag_offset=off)
    assert d['tag_offset'] == pytest.approx(list(off)) and d0['tag_offset'] == [0.0, 0.0, 0.0]
    truth = truth_error_mm(s0, cam, truth_poses(['O1-K1'], ks), off)
    assert abs(d['D_H_mm'] - d0['D_H_mm']) > 0.02
    assert d['D_mm']['h0_to_h1']['fit']['max'] == pytest.approx(truth, abs=0.05)
    s0o = dict(s0, tag_offset=list(off))                       # default: the offset of session H0
    assert cd.compare(s0o, s1, ['O1-K1'], ks)['D_H_mm'] == pytest.approx(d['D_H_mm'])


@pytest.mark.parametrize('bad', [dict(f=0.0), dict(f=float('nan')), dict(f=-900.0), dict(k=[None, 0.0])])
def test_bad_camera_numbers_refused(s0, bad):
    """CD3-4: f = 0 divided by zero (exit 1); now ValueError."""
    s1 = json.loads(json.dumps(s0))
    s1['camera'].update(bad)
    with pytest.raises(ValueError):
        cd.compare(s0, s1, FIG)


def test_distortion_change_trips_the_threshold(s0):
    d = cd.compare(s0, make_session(warped(0.03)), FIG)
    assert d['vuot'] and d['D_H_mm'] > cd.EPS_MM


def test_d_h_is_the_largest_of_both_directions_and_hypotheses(s0):
    d = cd.compare(s0, make_session(warped(-0.03)), FIG)
    vals = [v['max'] for w in d['D_mm'].values() for v in w.values()]
    a = max(v['max'] for v in d['D_mm']['h0_to_h1'].values())
    b = max(v['max'] for v in d['D_mm']['h1_to_h0'].values())
    assert abs(a - b) > 0.02
    assert d['D_H_mm'] == pytest.approx(max(vals))


def test_repeat_noise_raises_the_threshold_and_changes_the_decision(s0):
    s1 = make_session(warped(0.01))
    d = cd.compare(s0, s1, FIG)
    assert cd.EPS_MM < d['D_H_mm'] < 2 * cd.EPS_MM and d['vuot']
    d_lap = d['D_H_mm'] / 2 + 0.2                            # 2 x D_lap above D_H
    d2 = cd.compare(s0, s1, FIG, d_lap_mm=d_lap)
    assert d2['eps_H_mm'] == pytest.approx(2 * d_lap) and not d2['vuot']
    d3 = cd.compare(s0, s1, FIG, d_lap_mm=0.4)
    assert d3['eps_H_mm'] == pytest.approx(cd.EPS_MM) and d3['vuot']


def test_k_lanes_are_scored_when_run(s0):
    s1 = make_session(warped(0.001))
    f = cd.compare(s0, s1, FIG)
    k = cd.compare(s0, s1, FIG + ['O1-K1'], K_NEAR)
    assert k['lanes'] == ['O1-F', 'O1-K1'] and k['D_H_mm'] > f['D_H_mm'] + 0.3
    assert not f['vuot'] and k['vuot']


def test_moved_floor_mark_is_named(s0):
    tm = file_marks()
    x, y, side = tm[14]
    tm[14] = (x + 0.015, y - 0.010, side)                   # mark 14 kicked by 18 mm
    d = cd.compare(s0, make_session(truth_camera(), true_marks=tm), FIG)
    assert [m['id'] for m in d['marks_grown']] == [14]
    assert d['shift_px']['max_id'] == 14


def test_missing_mark_in_h1_is_listed(s0):
    d = cd.compare(s0, make_session(truth_camera(), drop=(24,)), FIG)
    assert d['missing_in_h1'] == [24] and d['D_H_mm'] < 0.5


def test_missing_h1(s0):
    d = cd.compare(s0, None, FIG)
    assert d['thieu_H1'] and d['vuot'] and d['D_H_mm'] is None and d['s1_session_sha256'] is None


@pytest.mark.parametrize('key,val', [('zc_tape_m', 2.95), ('plumb_xy_m', [0.04, -0.02]), ('tag_h_m', 0.13),
                                     ('car_side_m', 0.150), ('width', 3840),
                                     ('markers_csv', [dict(file='moc_san_do.csv', sha256='n' * 64)]),
                                     ('markers_csv', [])])
def test_sessions_of_another_set_up_refused(s0, key, val):
    s1 = make_session(truth_camera(), **{key: val})
    with pytest.raises(ValueError):
        cd.compare(s0, s1, FIG)


def test_different_mark_coordinates_refused(s0):
    s1 = json.loads(json.dumps(s0))
    s1['marks']['3'][0] += 0.001
    with pytest.raises(ValueError):
        cd.compare(s0, s1, FIG)


@pytest.mark.parametrize('kw', [dict(eps_mm=float('nan')), dict(eps_mm=0.0), dict(eps_mm=-1.0),
                                dict(d_lap_mm=float('nan')), dict(d_lap_mm=-0.1), dict(mark_px=float('inf'))])
def test_bad_numbers_refused(s0, kw):
    with pytest.raises(ValueError):
        cd.compare(s0, json.loads(json.dumps(s0)), FIG, **kw)


def test_session_without_corners_refused(tmp_path, s0):
    old = {k: v for k, v in s0.items() if k not in ('ref_corners_px', '_sha256')}
    p = tmp_path / 'old.json'
    p.write_text(json.dumps(old), encoding='utf-8')
    with pytest.raises(ValueError):
        cd.load_session(str(p))


# ---------------------------------------------------------------- combine
KS_K = [1.0, 0.0, -math.pi / 2]


def sc(outcome, frames='f0', esp='e' * 64, t0=1000, off=0.25, U=0.005, ks=(0.0, 0.0, 0.785), task='O1-F05'):
    return dict(outcome=outcome, outcome_reasons=[], conditions={}, e_perp_max=0.01, completed='true', task=task,
                t0_log_us=t0, offset_s=off, U_m=U, D=0.1, k_start=list(ks),
                thresholds=dict(t_miss=0.25, status='khoa o C8'),
                frames_sha256=frames, esp32_sha256=esp, frames_valid=600, gap_max=0.02, shift_max_px=0.5)


def tr(session='s0', video='r' * 64, frames='f0', off=(0.0, 0.0, 0.0), n=640, shape=((1080, 1920),)):
    return dict(session_sha256=session, video_sha256=video, frames_sha256=frames, tag_offset=list(off), frames=n,
                decoded_shapes=[list(s) for s in shape])


TASKS = ['O1-F02', 'O1-F05', 'O1-K1']
DRIFT_OK = dict(kind='c7_calib_drift', vuot=False, thieu_H1=False, D_H_mm=0.5, eps_H_mm=2.0,
                s0_session_sha256='s0', s1_session_sha256='s1', tasks=TASKS, k_start=KS_K, tag_offset=[0.0, 0.0, 0.0])
DRIFT_OVER = dict(DRIFT_OK, vuot=True, D_H_mm=3.1)
DRIFT_MISS = dict(DRIFT_OK, vuot=True, thieu_H1=True, D_H_mm=None, s1_session_sha256=None)


def over(o0, o1, **k1):
    return cd.combine(DRIFT_OVER, sc(o0), tr('s0'), 'f0', sc(o1, frames='f1', **k1), tr('s1', frames='f1'), 'f1')


def expected(o0, o1):
    if 'hong_he_do' in (o0, o1):
        return 'hong_he_do'
    return o0 if o0 == o1 else 'chua_ket_luan'


@pytest.mark.parametrize('o0,o1', list(itertools.product(cd.OUTCOMES, cd.OUTCOMES)))
def test_combine_table_all_16(o0, o1):
    r = over(o0, o1)
    assert r['outcome'] == expected(o0, o1)
    split = o0 != o1 and 'hong_he_do' not in (o0, o1)
    assert 'calib_drift' in r['calib_drift']['flags']
    assert ('calib_drift_split' in r['calib_drift']['flags']) == split
    assert r['calib_drift']['outcome_h0'] == o0 and r['calib_drift']['outcome_h1'] == o1
    assert r['calib_drift']['h1_score']['outcome'] == o1                    # whole H1 score kept


def test_combine_below_threshold_keeps_h0():
    r = cd.combine(DRIFT_OK, sc('dat'), tr('s0'), 'f0')
    assert r['outcome'] == 'dat' and r['calib_drift']['flags'] == []
    with pytest.raises(ValueError):
        cd.combine(DRIFT_OK, sc('dat'), tr('s0'), 'f0', sc('khong_dat', frames='f1'), tr('s1', frames='f1'), 'f1')


@pytest.mark.parametrize('o0,final', [('dat', 'chua_ket_luan'), ('khong_dat', 'chua_ket_luan'),
                                      ('chua_ket_luan', 'chua_ket_luan'), ('hong_he_do', 'hong_he_do')])
def test_combine_missing_h1(o0, final):
    r = cd.combine(DRIFT_MISS, sc(o0), tr('s0'), 'f0')
    assert r['outcome'] == final and r['calib_drift']['flags'] == ['thieu_H1']
    with pytest.raises(ValueError):
        cd.combine(DRIFT_MISS, sc(o0), tr('s0'), 'f0', sc(o0, frames='f1'), tr('s1', frames='f1'), 'f1')


def _drop(d, k):
    return {kk: v for kk, v in d.items() if kk != k}


# each bad drift.json is given with the inputs its valid twin accepts (DRIFT_OK and DRIFT_MISS: H0 only;
# DRIFT_OVER: both scores), so only the defect can make it fail (CD2-7)
@pytest.mark.parametrize('bad,two', [
    (dict(DRIFT_OK, vuot=None), False), (_drop(DRIFT_OK, 'vuot'), False), (dict(DRIFT_OK, thieu_H1=0), False),
    (dict(DRIFT_MISS, thieu_H1='yes'), False),
    (dict(DRIFT_MISS, vuot=False), False), (dict(DRIFT_MISS, D_H_mm=3.0), False),
    (dict(DRIFT_OVER, vuot=False), False),                    # D_H above eps_H written as below: H0 only
    (dict(DRIFT_OK, vuot=True), True),                        # below written as above: both scores
    (dict(DRIFT_OK, eps_H_mm=float('nan')), False), (dict(DRIFT_OK, D_H_mm=float('nan')), False),
    (dict(DRIFT_OK, s1_session_sha256=None), False), (dict(DRIFT_OK, s0_session_sha256=''), False),
    (dict(DRIFT_OK, kind='x'), False),
    (_drop(DRIFT_OK, 'tasks'), False), (dict(DRIFT_OK, tasks=[]), False), (dict(DRIFT_OK, tasks=['O1-F05', 'O1-Z']), False),
    (dict(DRIFT_OK, k_start=None), False), (dict(DRIFT_OK, k_start=[1.0, 0.0]), False),
    (_drop(DRIFT_OK, 'D_H_mm'), False), (_drop(DRIFT_OK, 's1_session_sha256'), False), ([], False),
    (_drop(DRIFT_OK, 'tag_offset'), False), (dict(DRIFT_OK, tag_offset=None), False)])
def test_bad_drift_files_refused(bad, two):
    args = (sc('dat', frames='f1'), tr('s1', frames='f1'), 'f1') if two else ()
    with pytest.raises(ValueError):
        cd.combine(bad, sc('dat'), tr('s0'), 'f0', *args)


@pytest.mark.parametrize('case', ['no_second', 'session0', 'session1', 'video', 'frames0', 'frames1', 'esp32',
                                  'U', 'k_start', 't0', 'offset', 'offset_nan', 'outcome', 'task', 'tag_offset',
                                  'n_frames', 'shape', 'h0_copied_into_h1', 'score_frames', 'track_no_frames_sha',
                                  'no_D_both', 'no_thresholds_both', 'no_t0_both', 'no_k_start_both',
                                  'score_not_dict', 'track_not_dict', 'score_null', 'track_null',
                                  'D', 'threshold', 'frames_null_both', 'shapes_null_both', 'threshold_null_both',
                                  'frames_bool', 'shape_zero', 'offset_not_drift_both'])
def test_provenance_and_scoring_set_up_checked(case):
    a, ta, fa = sc('dat'), tr('s0'), 'f0'
    b, tb, fb = sc('dat', frames='f1'), tr('s1', frames='f1'), 'f1'
    if case == 'no_second':
        b = tb = fb = None
    elif case == 'session0':
        ta = tr('sX')
    elif case == 'session1':
        tb = tr('s0', frames='f1')                           # the H1 score made with the H0 session
    elif case == 'video':
        tb = tr('s1', video='q' * 64, frames='f1')
    elif case == 'frames0':
        fa = 'other'
    elif case == 'frames1':
        fb = 'f0'
    elif case == 'esp32':
        b = sc('dat', frames='f1', esp='z' * 64)
    elif case == 'U':
        b = sc('dat', frames='f1', U=0.0)
    elif case == 'k_start':
        b = sc('dat', frames='f1', ks=(0.1, 0.0, 0.785))
    elif case == 't0':
        b = sc('dat', frames='f1', t0=2000)
    elif case == 'offset':
        b = sc('dat', frames='f1', off=0.26)
    elif case == 'offset_nan':
        b = sc('dat', frames='f1', off=float('nan'))
    elif case == 'outcome':
        b = sc('xyz', frames='f1')
    elif case == 'task':                                     # CD2-4
        b = sc('dat', frames='f1', task='O1-F02')
    elif case == 'tag_offset':                               # CD2-4
        tb = tr('s1', frames='f1', off=(0.01, 0.0, 0.0))
    elif case == 'n_frames':
        tb = tr('s1', frames='f1', n=300)
    elif case == 'shape':
        tb = tr('s1', frames='f1', shape=((2160, 3840),))
    elif case == 'h0_copied_into_h1':                        # CD2-3: frames.csv and score of H0 put next
        b, fb = sc('dat', frames='f0'), 'f0'                 # to the H1 track.json (which wrote f1)
    elif case == 'score_frames':                             # track.json = the file next to it, score not
        b = sc('dat', frames='fX')
    elif case == 'track_no_frames_sha':
        tb = _drop(tr('s1', frames='f1'), 'frames_sha256')
    elif case.startswith('no_') and case.endswith('_both'):  # CD2-6: absent on both sides is not "same"
        k = {'no_D_both': 'D', 'no_thresholds_both': 'thresholds', 'no_t0_both': 't0_log_us',
             'no_k_start_both': 'k_start'}[case]
        a, b = _drop(a, k), _drop(b, k)
    elif case == 'score_not_dict':
        b = [b]
    elif case == 'track_not_dict':
        tb = [tb]
    elif case == 'score_null':                               # JSON null as the H0 score: ValueError
        a = None
    elif case == 'track_null':
        ta = None
    elif case == 'D':                                        # CD3-6: complete pair, only D differs
        b = dict(b, D=0.2)
    elif case == 'threshold':                                # CD3-6: only one threshold differs
        b = dict(b, thresholds=dict(t_miss=0.30, status='khoa o C8'))
    elif case == 'frames_null_both':                         # CD3-5: null on both sides is not "same"
        ta, tb = dict(ta, frames=None), dict(tb, frames=None)
    elif case == 'shapes_null_both':
        ta, tb = dict(ta, decoded_shapes=None), dict(tb, decoded_shapes=None)
    elif case == 'threshold_null_both':
        a = dict(a, thresholds=dict(t_miss=None, status='x'))
        b = dict(b, thresholds=dict(t_miss=None, status='x'))
    elif case == 'frames_bool':
        ta, tb = dict(ta, frames=True), dict(tb, frames=True)
    elif case == 'shape_zero':
        ta, tb = dict(ta, decoded_shapes=[[0, 1920]]), dict(tb, decoded_shapes=[[0, 1920]])
    elif case == 'offset_not_drift_both':                    # CD3-2: both tracks agree, the drift does not
        ta, tb = tr('s0', off=(0.03, 0.0, 0.0)), tr('s1', frames='f1', off=(0.03, 0.0, 0.0))
    with pytest.raises(ValueError):
        cd.combine(DRIFT_OVER, a, ta, fa, b, tb, fb)


def test_both_scores_complete_are_accepted():
    """Control of the case table above: the unchanged pair goes through."""
    r = cd.combine(DRIFT_OVER, sc('dat'), tr('s0'), 'f0', sc('dat', frames='f1'), tr('s1', frames='f1'), 'f1')
    assert r['outcome'] == 'dat'


@pytest.mark.parametrize('drift', [DRIFT_OK, DRIFT_MISS, DRIFT_OVER])
def test_task_outside_the_drift_refused_in_every_branch(drift):
    """CD2-5: the drift was computed for other lanes; below eps_H too."""
    d = dict(drift, tasks=['O1-F02', 'O1-F08'])
    two = (sc('dat', frames='f1'), tr('s1', frames='f1'), 'f1') if drift['vuot'] and not drift['thieu_H1'] else ()
    cd.combine(drift, sc('dat'), tr('s0'), 'f0', *two)                     # O1-F05 is in TASKS: accepted
    with pytest.raises(ValueError):
        cd.combine(d, sc('dat'), tr('s0'), 'f0', *two)


@pytest.mark.parametrize('drift', [DRIFT_OK, DRIFT_OVER])
def test_k_start_of_a_k_score_must_be_that_of_the_drift(drift):
    """CD2-5: K1 scored on another nominal start pose than the one the drift was sampled on."""
    two = drift['vuot']
    good = sc('dat', task='O1-K1', ks=KS_K)
    bad = sc('dat', task='O1-K1', ks=(1.0, 0.1, -math.pi / 2))
    args = lambda s: (dict(s, frames_sha256='f1'), tr('s1', frames='f1'), 'f1') if two else ()   # noqa: E731
    cd.combine(drift, good, tr('s0'), 'f0', *args(good))
    with pytest.raises(ValueError):
        cd.combine(drift, bad, tr('s0'), 'f0', *args(bad))


# ---------------------------------------------------------------- CLI
def _session_file(tmp_path, name, s):
    p = tmp_path / f'{name}.json'
    p.write_text(json.dumps({k: v for k, v in s.items() if k != '_sha256'}), encoding='utf-8')
    return str(p)


def _run_dir(tmp_path, name, session_path, score, extra_score=''):
    d = tmp_path / name
    d.mkdir()
    (d / 'frames.csv').write_text(f'idx,t_s\n0,{name}\n', encoding='utf-8')
    fsha = cm.sha256(str(d / 'frames.csv'))
    t = dict(tr(cm.sha256(session_path), frames=fsha))
    (d / 'track.json').write_text(json.dumps(t), encoding='utf-8')
    text = json.dumps(dict(score, frames_sha256=fsha))
    if extra_score:
        text = text[:-1] + ', ' + extra_score + '}'
    (d / 'score.json').write_text(text, encoding='utf-8')
    return str(d / 'score.json'), str(d / 'track.json')


def test_cli_round_trip(tmp_path, s0, capsys):
    p0 = _session_file(tmp_path, 's0', s0)
    p1 = _session_file(tmp_path, 's1', make_session(warped(0.03)))
    dr = str(tmp_path / 'drift.json')
    assert cd.main(['so', '--s0', p0, '--s1', p1, '--tasks', 'O1-F05', '--out', dr]) == 0
    assert capsys.readouterr().out.rstrip().endswith('#ket_qua,ma=0,vuot=1')
    assert cd.main(['so', '--s0', p0, '--s1', p1, '--tasks', 'O1-F05', '--out', dr]) == 2   # no overwrite
    capsys.readouterr()
    sa, ta = _run_dir(tmp_path, 'r0', p0, sc('dat'))
    sb, tb = _run_dir(tmp_path, 'r1', p1, sc('khong_dat'))
    out = str(tmp_path / 'final.json')
    assert cd.main(['gop', '--drift', dr, '--s0-score', sa, '--s0-track', ta, '--s1-score', sb,
                    '--s1-track', tb, '--out', out]) == 0
    assert capsys.readouterr().out.rstrip().endswith('#ket_qua,ma=0,outcome=chua_ket_luan')
    r = json.loads(open(out, encoding='utf-8').read())
    assert r['outcome'] == 'chua_ket_luan' and r['calib_drift']['h1_score']['outcome'] == 'khong_dat'
    # the H1 run scored with the H0 session: refused
    sc_, tc = _run_dir(tmp_path, 'r1b', p0, sc('dat'))
    assert cd.main(['gop', '--drift', dr, '--s0-score', sa, '--s0-track', ta, '--s1-score', sc_,
                    '--s1-track', tc, '--out', str(tmp_path / 'x.json')]) == 2
    assert capsys.readouterr().out.rstrip().endswith('#ket_qua,ma=2')
    miss = str(tmp_path / 'miss.json')
    assert cd.main(['so', '--s0', p0, '--tasks', 'O1-F05', '--out', miss]) == 0
    assert capsys.readouterr().out.rstrip().endswith('#ket_qua,ma=0,vuot=1')


@pytest.mark.parametrize('args', [['--tasks', 'O1-K1'], ['--tasks', 'O1-K1', '--k-start', '0,0'],
                                  ['--tasks', 'O1-K1', '--k-start', 'a,b,c'], ['--tasks', 'O1-F05', '--eps-mm', 'nan'],
                                  ['--tasks', 'O1-Z'], ['--tasks', 'O1-F05', '--tag-offset', '0.01,0'],
                                  ['--tasks', 'O1-F05', '--tag-offset', '0,0,inf']])
def test_cli_bad_arguments_exit_2(tmp_path, s0, capsys, args):
    p0 = _session_file(tmp_path, 's0', s0)
    assert cd.main(['so', '--s0', p0, '--out', str(tmp_path / 'd.json')] + args) == 2
    assert capsys.readouterr().out.rstrip().endswith('#ket_qua,ma=2')


def test_cli_tag_offset_and_zero_focal(tmp_path, s0, capsys):
    p0 = _session_file(tmp_path, 's0', s0)
    dr = str(tmp_path / 'd.json')
    assert cd.main(['so', '--s0', p0, '--s1', p0, '--tasks', 'O1-F05', '--tag-offset', '0.03,0,1', '--out', dr]) == 0
    d = json.loads(open(dr, encoding='utf-8').read())
    assert d['tag_offset'] == pytest.approx([0.03, 0.0, math.radians(1.0)])
    capsys.readouterr()
    bad = json.loads(json.dumps(s0))
    bad['camera']['f'] = 0.0                                   # CD3-4: was ZeroDivisionError, exit 1
    p1 = _session_file(tmp_path, 's1', bad)
    assert cd.main(['so', '--s0', p0, '--s1', p1, '--tasks', 'O1-F05', '--out', str(tmp_path / 'e.json')]) == 2
    assert capsys.readouterr().out.rstrip().endswith('#ket_qua,ma=2')


def _gop_below(tmp_path, s0, drift_text, extra_score=''):
    p0 = _session_file(tmp_path, 's0', s0)
    dr = tmp_path / 'drift.json'
    dr.write_text(drift_text.replace('S0SHA', cm.sha256(p0)), encoding='utf-8')
    sa, ta = _run_dir(tmp_path, 'r0', p0, sc('dat'), extra_score)
    return cd.main(['gop', '--drift', str(dr), '--s0-score', sa, '--s0-track', ta, '--out', str(tmp_path / 'o.json')])


GOOD_DRIFT = json.dumps(dict(DRIFT_OK, s0_session_sha256='S0SHA'))


def test_cli_gop_control_accepts_complete_files(tmp_path, s0, capsys):
    assert _gop_below(tmp_path, s0, GOOD_DRIFT) == 0
    assert capsys.readouterr().out.rstrip().endswith('#ket_qua,ma=0,outcome=dat')


@pytest.mark.parametrize('where', ['drift', 'score'])
@pytest.mark.parametrize('const', ['NaN', 'Infinity', '-Infinity'])
def test_cli_nan_in_json_refused(tmp_path, s0, capsys, where, const):
    """CD2-7: complete, valid files (the control above passes) with a non-finite constant in a field
    nothing else reads: only the JSON parser can refuse it."""
    drift = GOOD_DRIFT[:-1] + f', "note_x": {const}}}' if where == 'drift' else GOOD_DRIFT
    extra = f'"note_x": {const}' if where == 'score' else ''
    assert _gop_below(tmp_path, s0, drift, extra) == 2
    assert capsys.readouterr().out.rstrip().endswith('#ket_qua,ma=2')


@pytest.mark.parametrize('text', ['[]', '"x"', '3', 'null'])
def test_cli_json_not_an_object_exit_2(tmp_path, s0, capsys, text):
    """CD2-6: valid JSON that is not an object ends with code 2, not a traceback."""
    assert _gop_below(tmp_path, s0, text) == 2
    assert capsys.readouterr().out.rstrip().endswith('#ket_qua,ma=2')
