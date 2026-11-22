"""Tests of c7_u_moc khung (CAM design v4 section 5, can moc dich v3 section 2). SYNTHETIC detections projected from a
known camera (test_c7_33_moc helpers), NOT a measurement; D and the car pose are test numbers."""
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
import c7_measure as cm  # noqa: E402
import c7_u_moc as um  # noqa: E402
from test_c7_33_moc import make_session, run_frames, CAR_SIDE  # noqa: E402

D_TEST = 0.0017
GATES = dict(cm.DEFAULT_GATES)


@pytest.fixture(scope='module')
def sess():
    return make_session()


@pytest.fixture(scope='module')
def run(sess):
    frames = run_frames(20, fps=60.0, noise=0.0, car=(0.2, -0.1, math.radians(20.0)))   # >= scale_min neighbours
    rows, _, _ = cm.track_chains(frames, 'syn', 'V', sess, tag_offset=(0.0, 0.0, 0.0))
    return frames, rows


def _cfg(sess, frames, tmp_path):
    p = tmp_path / 'dets.npz'
    cm.save_dets(str(p), frames, 'V', (10, 10))
    return dict(session_obj=sess, dets=str(p), a6=None, D=D_TEST, gates=GATES, dets_sha='DS', lich_sha='LS', run='r1')


def _task(rows, k):
    r = rows[k]
    return (k, dict(tag_x=r['tag_x'], tag_y=r['tag_y'], tag_psi=r['psi'], moc_neo_xe=r['moc_neo_xe'],
                    scale_ref=r['scale_ref']))


def test_seed_is_fixed_by_the_inputs_and_differs_per_frame():
    a = um.seed_of('d', 'l', 'r', 5)
    assert a == um.seed_of('d', 'l', 'r', 5)
    assert len({a, um.seed_of('d', 'l', 'r', 6), um.seed_of('d', 'l', 'r2', 5), um.seed_of('d', 'l2', 'r', 5),
                um.seed_of('d2', 'l', 'r', 5)}) == 5


def test_combinations_lie_in_the_disc_half_on_its_edge():
    s = um.combos(np.random.default_rng(1), 30, D_TEST)
    assert s.shape == (um.N_COMB, 30, 2)
    r = np.hypot(s[..., 0], s[..., 1])
    assert r.max() <= D_TEST * (1 + 1e-12)
    edge = np.isclose(r, D_TEST, rtol=0, atol=1e-15).mean()
    assert 0.45 < edge < 0.55


def test_frame_bound_matches_the_linear_sensitivity(sess, run, tmp_path):
    """The 8-direction sum S_t lies between cos(22.5 deg) and 1 times the linear bound of c7_measure.car_sensitivity
    (sum of spectral norms x D, tag offset 0 so the wheel centre is the tag centre); the problem is near linear at D."""
    frames, rows = run
    um._init(_cfg(sess, frames, tmp_path))
    k = 10
    out = um.khung_mot(_task(rows, k))
    assert out['loi'] == '' and out['n_neo'] == len(str(rows[k]['moc_neo_xe']).split(';'))
    ctx = cm.ChainCtx(sess)
    _, _, d = frames[k]
    anchors = sorted(int(x) for x in str(rows[k]['moc_neo_xe']).split(';'))
    cam = ctx.pose(cm.CHAIN_XE, anchors, d)[0]
    lin = cm.car_sensitivity(cam, anchors, d, ctx.marks, d[cm.CAR_ID], sess['car_side_m'], sess['kappa'],
                             sess['plumb_xy_m'], (0.0, 0.0, 0.0), D_TEST)
    assert math.cos(math.radians(22.5)) * 0.97 * lin['U_xy'] <= out['S_t_m'] <= 1.03 * lin['U_xy']
    assert math.cos(math.radians(22.5)) * 0.97 * lin['U_psi'] <= out['S_psi_rad'] <= 1.03 * lin['U_psi']
    assert out['C_t_m'] <= out['S_t_m'] and out['C_psi_rad'] <= out['S_psi_rad']
    assert out['S_sin'] <= out['S_psi_rad'] / 2 + 1e-15
    assert out['cong_goc'] and out['doi_cong_cuc_dai'] == []


def test_same_seed_same_numbers_and_a_mismatch_blocks_the_frame(sess, run, tmp_path):
    frames, rows = run
    um._init(_cfg(sess, frames, tmp_path))
    a, b = um.khung_mot(_task(rows, 2)), um.khung_mot(_task(rows, 2))
    assert a == b
    k, r = _task(rows, 2)
    r = dict(r, tag_x=r['tag_x'] + 1e-4)                      # frames.csv not reproduced by the refit
    assert um.khung_mot((k, r))['loi'] == 'khop_goc_khac_frames'
    k, r = _task(rows, 2)
    r = dict(r, moc_neo_xe=r['moc_neo_xe'] + ';99')           # an anchor not detected in the frame
    assert um.khung_mot((k, r))['loi'] == 'thieu_phat_hien'


def test_cli_refuses_a_schedule_not_converged(sess, run, tmp_path, monkeypatch):
    frames, rows = run
    sp = tmp_path / 's.json'
    sp.write_text(json.dumps(cm._json(sess)), encoding='utf-8')
    dp = tmp_path / 'dets.npz'
    cm.save_dets(str(dp), frames, 'V', (10, 10))
    fp = tmp_path / 'frames.csv'
    cm.write_frames(str(fp), rows)
    (tmp_path / 'track.json').write_text(json.dumps(dict(tag_offset=[0.0, 0.0, 0.0])), encoding='utf-8')
    for n, o in (('u.json', dict(x=1)), ('lich.json', dict(lich_sha256='L'))):
        (tmp_path / n).write_text(json.dumps(o), encoding='utf-8')
    hs = tmp_path / 'hs.json'
    hs.write_text(json.dumps(dict(cong_khung=GATES)), encoding='utf-8')
    args = ['khung', '--frames', str(fp), '--dets', str(dp), '--session', str(sp), '--u-so', str(tmp_path / 'u.json'),
            '--lich', str(tmp_path / 'lich.json'), '--run', 'r1', '--hang-so', str(hs), '--out', str(tmp_path / 'o')]
    monkeypatch.setattr(cm, 'verify_final', lambda *x, **k: ['not converged'])
    assert um.main(args) == 2
    monkeypatch.setattr(cm, 'verify_final', lambda *x, **k: [])
    monkeypatch.setattr(cm, 'undetected_bound', lambda u: D_TEST)
    assert um.main(args + ['--tu-s', '0.03', '--den-s', '0.05']) == 0
    lines = (tmp_path / 'o' / 'u_moc_khung.jsonl').read_text(encoding='utf-8').splitlines()
    info = json.loads(lines[0])
    assert info['format'] == um.FORMAT_KHUNG and info['bo_lay_mau'] == um.SAMPLER and info['n_loi'] == 0
    assert info['n_khung'] == len(lines) - 1 == 2 and info['huong_ma'] == 'tag_psi'     # frames.csv of c7_measure since 10043dc
    assert info['car_pose']['side_m'] == CAR_SIDE


def test_only_gate_2a_is_a_dependence(sess, run, tmp_path):
    """The floor-fit rms under an alternative map is not a dependence (the observed rms does not move): it is
    reported only. Gate (2a) res_sim is: with car_res_sim_m set just above the base value, the worst poses flip it."""
    frames, rows = run
    cfg = _cfg(sess, frames, tmp_path)
    cfg['gates'] = dict(GATES, rms_extra_px=0.0)
    um._init(cfg)
    out = um.khung_mot(_task(rows, 10))
    assert out['loi'] == '' and out['rms_max_px'] > out['rms_goc_px'] and out['doi_cong_cuc_dai'] == []
    base_res = min(out['xau_xy']['res_sim_m'], out['xau_psi']['res_sim_m'])
    cfg['gates'] = dict(GATES, car_res_sim_m=base_res * 0.999999)
    um._init(cfg)
    out2 = um.khung_mot(_task(rows, 10))
    assert out2['doi_cong_bat_ky'] > 0
