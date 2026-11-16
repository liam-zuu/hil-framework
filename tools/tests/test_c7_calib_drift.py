"""c7_calib_drift.py: compare the clips H of a session, score twice (O1 v11.28 section 4; design
CLAUDE_THESIS 03_CODE/docs/2026-11-15_Thiet_ke_so_clip_H_dau_cuoi_v1.md; Codex Sol round 1 CD1-2..11).
Sessions are synthetic: a truth camera sees the TRUE mark positions, the session is fitted to the mark
FILE (as calibrate does)."""
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
from test_c7_clip_probe import MARKS, SIDES, W, H, truth_camera  # noqa: E402

TAG_H, ZC_TAPE, PLUMB = 0.12, 2.90, [0.03, -0.02]
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
             zc_tape_m=ZC_TAPE, plumb_xy_m=list(PLUMB), tag_h_m=TAG_H, car_side_m=0.1445, width=W, height=H,
             kappa=TAG_H / ZC_TAPE, markers_csv=[dict(file='moc_san_do.csv', sha256='m' * 64)], _sha256=sha)
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


@pytest.fixture(scope='module')
def s0():
    return make_session(truth_camera())


# ---------------------------------------------------------------- lanes (CD1-8, CD1-9)
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


def test_lane_samples_every_5_cm_and_keep_the_end():
    s = np.linspace(0.0, 1.03, 1031)
    q = cd._resample(s, s, np.zeros_like(s), 0.05)
    d = np.diff(q[:, 0])
    assert np.allclose(d[:-1], 0.05) and d[-1] == pytest.approx(0.03) and q[-1, 0] == pytest.approx(1.03)


# ---------------------------------------------------------------- compare
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
    assert d['D_H_mm'] == pytest.approx(expect, rel=0.25)
    assert not d['vuot'] and d['marks_grown'] == []
    assert np.hypot(*d['camera_shift_m'][:2]) == pytest.approx(math.hypot(0.02, 0.01), rel=0.1)


def test_camera_height_change_enters_through_kappa(s0):
    # the Z_c prior of the fit (tape, sigma 20 mm) holds part of a height change, so only the direction
    # and a clear rise above the unchanged case are asserted
    calm = cd.compare(s0, make_session(truth_camera()), FIG)
    d = cd.compare(s0, make_session(moved_camera((0.0, 0.0, 0.10), 0.0)), FIG)
    assert d['kappa'][1] < d['kappa'][0]
    assert d['D_H_mm'] > 10 * max(calm['D_H_mm'], 0.01)


def test_focal_length_change_is_seen(s0):
    c = truth_camera()
    zoom = cp.Camera(c.model, c.f * 1.02, c.cx, c.cy, c.k, c.rvec, c.tvec)
    calm = cd.compare(s0, make_session(truth_camera()), FIG)
    d = cd.compare(s0, make_session(zoom), FIG)
    assert d['focal_rel'] == pytest.approx(0.02, abs=0.005)
    assert d['D_H_mm'] > 10 * max(calm['D_H_mm'], 0.01)


def test_distortion_change_trips_the_threshold(s0):
    d = cd.compare(s0, make_session(warped(0.03)), FIG)
    assert d['vuot'] and d['D_H_mm'] > cd.EPS_MM


def test_d_h_is_the_larger_of_both_directions(s0):
    d = cd.compare(s0, make_session(warped(-0.03)), FIG)
    a, b = d['D_mm']['h0_to_h1']['max'], d['D_mm']['h1_to_h0']['max']
    assert abs(a - b) > 0.02
    assert d['D_H_mm'] == pytest.approx(max(a, b))


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
    k = cd.compare(s0, s1, FIG + ['O1-K1'], (1.0, 0.0, math.radians(-90)))
    assert k['lanes'] == ['O1-F', 'O1-K1'] and k['D_H_mm'] >= f['D_H_mm']


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
def sc(outcome, frames='f0', esp='e' * 64, t0=1000, off=0.25, U=0.005, ks=(0.0, 0.0, 0.785)):
    return dict(outcome=outcome, outcome_reasons=[], conditions={}, e_perp_max=0.01, completed='true',
                t0_log_us=t0, offset_s=off, U_m=U, D=0.1, k_start=list(ks), thresholds=dict(t_miss=0.25),
                frames_sha256=frames, esp32_sha256=esp, frames_valid=600, gap_max=0.02, shift_max_px=0.5)


def tr(session='s0', video='r' * 64):
    return dict(session_sha256=session, video_sha256=video)


DRIFT_OK = dict(kind='c7_calib_drift', vuot=False, thieu_H1=False, D_H_mm=0.5, eps_H_mm=2.0,
                s0_session_sha256='s0', s1_session_sha256='s1')
DRIFT_OVER = dict(DRIFT_OK, vuot=True, D_H_mm=3.1)
DRIFT_MISS = dict(kind='c7_calib_drift', vuot=True, thieu_H1=True, D_H_mm=None, eps_H_mm=2.0,
                  s0_session_sha256='s0', s1_session_sha256=None)


def over(o0, o1, **k1):
    return cd.combine(DRIFT_OVER, sc(o0), tr('s0'), 'f0', sc(o1, frames='f1', **k1), tr('s1'), 'f1')


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
        cd.combine(DRIFT_OK, sc('dat'), tr('s0'), 'f0', sc('khong_dat', frames='f1'), tr('s1'), 'f1')


@pytest.mark.parametrize('o0,final', [('dat', 'chua_ket_luan'), ('khong_dat', 'chua_ket_luan'),
                                      ('chua_ket_luan', 'chua_ket_luan'), ('hong_he_do', 'hong_he_do')])
def test_combine_missing_h1(o0, final):
    r = cd.combine(DRIFT_MISS, sc(o0), tr('s0'), 'f0')
    assert r['outcome'] == final and r['calib_drift']['flags'] == ['thieu_H1']
    with pytest.raises(ValueError):
        cd.combine(DRIFT_MISS, sc(o0), tr('s0'), 'f0', sc(o0, frames='f1'), tr('s1'), 'f1')


@pytest.mark.parametrize('bad', [
    dict(DRIFT_OVER, vuot=None), {k: v for k, v in DRIFT_OVER.items() if k != 'vuot'},
    dict(DRIFT_MISS, vuot=False), dict(DRIFT_MISS, D_H_mm=3.0), dict(DRIFT_OVER, vuot=False),
    dict(DRIFT_OK, vuot=True), dict(DRIFT_OVER, eps_H_mm=float('nan')), dict(DRIFT_OVER, D_H_mm=float('nan')),
    dict(DRIFT_OVER, s1_session_sha256=None), dict(DRIFT_OVER, kind='x'), dict(DRIFT_OVER, thieu_H1='no')])
def test_bad_drift_files_refused(bad):
    with pytest.raises(ValueError):
        cd.combine(bad, sc('dat'), tr('s0'), 'f0', sc('dat', frames='f1'), tr('s1'), 'f1')


@pytest.mark.parametrize('case', ['no_second', 'session0', 'session1', 'video', 'frames0', 'frames1', 'esp32',
                                  'U', 'k_start', 't0', 'offset', 'offset_nan', 'outcome'])
def test_provenance_and_scoring_set_up_checked(case):
    a, ta, fa = sc('dat'), tr('s0'), 'f0'
    b, tb, fb = sc('dat', frames='f1'), tr('s1'), 'f1'
    if case == 'no_second':
        b = tb = fb = None
    elif case == 'session0':
        ta = tr('sX')
    elif case == 'session1':
        tb = tr('s0')                                        # the H1 score made with the H0 session
    elif case == 'video':
        tb = tr('s1', video='q' * 64)
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
    with pytest.raises(ValueError):
        cd.combine(DRIFT_OVER, a, ta, fa, b, tb, fb)


# ---------------------------------------------------------------- CLI
def _session_file(tmp_path, name, s):
    p = tmp_path / f'{name}.json'
    p.write_text(json.dumps({k: v for k, v in s.items() if k != '_sha256'}), encoding='utf-8')
    return str(p)


def _run_dir(tmp_path, name, session_path, score):
    d = tmp_path / name
    d.mkdir()
    (d / 'frames.csv').write_text(f'idx,t_s\n0,{name}\n', encoding='utf-8')
    (d / 'track.json').write_text(json.dumps(dict(session_sha256=cm.sha256(session_path), video_sha256='r' * 64)),
                                  encoding='utf-8')
    score = dict(score, frames_sha256=cm.sha256(str(d / 'frames.csv')))
    (d / 'score.json').write_text(json.dumps(score), encoding='utf-8')
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
                                  ['--tasks', 'O1-Z']])
def test_cli_bad_arguments_exit_2(tmp_path, s0, capsys, args):
    p0 = _session_file(tmp_path, 's0', s0)
    assert cd.main(['so', '--s0', p0, '--out', str(tmp_path / 'd.json')] + args) == 2
    assert capsys.readouterr().out.rstrip().endswith('#ket_qua,ma=2')


def test_cli_nan_in_json_refused(tmp_path, capsys):
    p = tmp_path / 'drift.json'
    p.write_text('{"kind": "c7_calib_drift", "vuot": false, "thieu_H1": false, "D_H_mm": NaN, "eps_H_mm": 2.0}',
                 encoding='utf-8')
    s = tmp_path / 's.json'
    s.write_text(json.dumps(sc('dat')), encoding='utf-8')
    t = tmp_path / 't.json'
    t.write_text(json.dumps(tr()), encoding='utf-8')
    (tmp_path / 'frames.csv').write_text('x\n', encoding='utf-8')
    assert cd.main(['gop', '--drift', str(p), '--s0-score', str(s), '--s0-track', str(t),
                    '--out', str(tmp_path / 'o.json')]) == 2
    assert capsys.readouterr().out.rstrip().endswith('#ket_qua,ma=2')
