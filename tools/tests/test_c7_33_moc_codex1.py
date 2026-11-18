"""Tests for the CAM-side findings of Codex round 1 on the 33-mark code (C33-1, 2, 3, 4, 5, 10, 15;
03_CODE/docs/review/raw/2026-11-18_code_33_moc_v1/tra_loi_codex.md). SYNTHETIC detections (NOT a measurement)."""
import csv
import json
import os
import sys

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
import c7_clip_probe as cp  # noqa: E402
import c7_measure as cm  # noqa: E402
import c7_moc_dich as md  # noqa: E402
from test_c7_clip_probe import W, H  # noqa: E402
from test_c7_33_moc import (MK, IDS, T, TR, dets, h_frames, make_session, fake_u_so, run_frames, series,  # noqa: E402
                            clip_of, sess, spike_run)  # noqa: F401  (fixtures)


def test_c33_2_base_change_after_an_occlusion_reaches_back_to_w_star():
    """Two marks shift 1.5 mm at 1.0 s, hidden 1.1-3.0 s, seen again shifted (Codex counterexample, 60 fps): the
    interval must start at their W* (about 0.6 s), not at the last base check (2.5 s)."""
    t, d = series(lambda tt: [0.0015 if tt >= 1.0 else 0.0, 0.0], 300)
    d[(t >= 1.1) & (t < 3.0)] = np.nan
    z = {k: np.zeros_like(d) for k in range(10, 20)}
    s = md.schedule([clip_of(t, {3: d, 4: d.copy(), **z})], T, TR, [3, 4] + list(range(10, 20)))
    assert s['loai'] == {} and len(s['nen_doi']) == 1
    assert t[s['nen_doi'][0]['tu'][1]] <= 1.0                      # covers the shifted samples 1.0-1.1 s
    assert md.inconclusive_at(s, 0, 62)[0]


def test_c33_10_clip_h_needs_every_mark_at_once():
    t, d = series(lambda tt: [0.0015 if tt >= 1.0 else 0.0, 0.0], 150)
    ids = list(range(20))
    th = 20.0 + np.arange(120) / 60.0
    hd = {k: np.zeros((120, 2)) for k in ids}
    for e in range(120):                                           # each mark hidden one frame in 20: 95 % each
        hd[e % 20][e] = np.nan
    run = {k: np.zeros_like(d) for k in ids}
    run[0], run[1] = d, d.copy()
    s = md.schedule([clip_of(t, run, 'r'), clip_of(th, hd, 'H_m', 'H')], T, TR, ids)
    assert len(s['nen_doi']) == 1 and s['nen_doi'][0]['den'] is None   # 95 % seen each, 0 % together: not closed
    hd2 = {k: np.zeros((120, 2)) for k in ids}
    s = md.schedule([clip_of(t, run, 'r'), clip_of(th, hd2, 'H_m', 'H')], T, TR, ids)
    assert s['nen_doi'][0]['den'] == [1, 0]


def test_c33_3_flag_uses_the_height_of_the_start_camera(sess):
    """A camera 0.3 m higher than the all-mark camera is trusted on its own data: the reference height is that of
    the start camera itself (with the all-mark camera's height it fails the 0.2 m tolerance)."""
    ctx = cm.ChainCtx(sess)
    c0 = ctx.cams['xe']
    R = cv2.Rodrigues(np.asarray(c0.rvec, float))[0]
    C = c0.centre + np.array([0.0, 0.0, 0.3])
    cam_hi = cp.Camera(c0.model, c0.f, c0.cx, c0.cy, c0.k, c0.rvec, -R @ C)
    d = dets(cam=cam_hi)
    obj = np.vstack([cp.mark_corners_floor(*MK[i]) for i in IDS])
    img = np.vstack([d[i] for i in IDS])
    assert cm.refit_pose_checked(cam_hi, obj, img)[2]
    assert not cm.refit_pose_checked(cam_hi, obj, img, abs(c0.centre[2]))[2]


def test_c33_4_cache_of_another_clip_with_the_same_name_is_not_used(tmp_path, sess, spike_run):
    _, frames = spike_run
    ctx = cm.ChainCtx(sess)
    c0 = ctx.cams['xe']                                            # clip B: the camera turned 0.3 deg (bumped)
    Rb = cv2.Rodrigues(np.array([0.0, 0.0, np.radians(0.3)]))[0] @ cv2.Rodrigues(np.asarray(c0.rvec, float))[0]
    cam_b = cp.Camera(c0.model, c0.f, c0.cx, c0.cy, c0.k, cv2.Rodrigues(Rb)[0].ravel(), -Rb @ c0.centre)
    rng = np.random.default_rng(9)
    other = [(k, k / 30.0, dets(cam=cam_b, noise=0.02, rng=rng, moved={11: (0.004, 0.0)} if k == 20 else None))
             for k in range(60)]
    L0a = cm.build_lich(sess, 'S', [('r1', 'luot', frames, 'DA')], cache_dir=str(tmp_path))
    cm.build_lich(sess, 'S', [('r1', 'luot', frames, 'DA')], fake_u_so(ctx, tap=L0a['tap_moc']), 'U', L0a,
                  cache_dir=str(tmp_path))                         # fills the cache with clip A
    L0b = cm.build_lich(sess, 'S', [('r1', 'luot', other, 'DB')])
    ub = fake_u_so(ctx, tap=L0b['tap_moc'])
    with_cache = cm.build_lich(sess, 'S', [('r1', 'luot', other, 'DB')], ub, 'U', L0b, cache_dir=str(tmp_path))
    no_cache = cm.build_lich(sess, 'S', [('r1', 'luot', other, 'DB')], ub, 'U', L0b)
    assert with_cache['lich_sha256'] == no_cache['lich_sha256']
    assert [11, 0, 20] in no_cache['chuoi']['xe']['moc_nghi']


def test_c33_5_u_so_of_another_session_or_setup_is_refused(sess, spike_run):
    _, frames = spike_run
    ctx = cm.ChainCtx(sess)
    runs = [('r1', 'luot', frames, 'D')]
    L0 = cm.build_lich(sess, 'S', runs)
    with pytest.raises(ValueError, match='another H_a'):
        cm.build_lich(sess, 'S', runs, fake_u_so(ctx, tap=L0['tap_moc'], ha='OTHER'), 'U', L0)
    u = fake_u_so(ctx, tap=L0['tap_moc'])
    u['chuoi']['xe']['nguong_m'] = 0.0015                          # not 1 mm + u_so
    with pytest.raises(ValueError, match='1 mm'):
        cm.build_lich(sess, 'S', runs, u, 'U', L0)
    u = fake_u_so(ctx, tap=L0['tap_moc'])
    u['model'] = 'B1'
    with pytest.raises(ValueError, match='model'):
        cm.build_lich(sess, 'S', runs, u, 'U', L0)
    u = fake_u_so(ctx, tap=L0['tap_moc'])
    u['nhom'] = u['nhom'][::-1]
    with pytest.raises(ValueError, match='groups'):
        cm.build_lich(sess, 'S', runs, u, 'U', L0)


def test_c33_15_floor_mark_missing_from_clip_h_blocks_the_session():
    fr = h_frames()
    for d in fr:
        d.pop(14)                                                  # never seen in clip H
    s = make_session(MK, fr)
    kc = s['kiem_cheo']
    assert 14 in [i for g in kc['nhom'] for i in g]                # still in its group
    assert ['xe', 14] in kc['nen_thieu']
    with pytest.raises(ValueError, match='baseline'):
        cm.ChainCtx(s)


def test_c33_1_final_pass_reports_not_concluded(tmp_path, sess):
    j = 6
    ctx = cm.ChainCtx(sess)
    # two marks of different groups shift 1.5 mm together (under T_raw: no moc_nghi): chain xe and the chains that see
    # both keep a base change to the end; chain 0 / chain 1 see one of them alone and exclude it
    frames = run_frames(90, moved_at=lambda k: {j: (0.0015, 0.0), 9: (0.0, 0.0015)} if k >= 30 else {},
                        car=(0.2, -0.1, 0.3))
    sp = tmp_path / 'session.json'
    sp.write_text(json.dumps(cm._json(sess)), encoding='utf-8')
    d1 = tmp_path / 'r1'
    d1.mkdir()
    cm.save_dets(str(d1 / 'dets.npz'), frames, 'V', (H, W))
    base = ['--session', str(sp), '--luot', f'r1={d1}']
    assert cm.main(['lich'] + base + ['--out', str(tmp_path / 'L0')]) == 0
    prev, k = 'L0', 0
    for k in range(1, 6):
        tap = json.loads((tmp_path / prev / 'tap_moc.json').read_text(encoding='utf-8'))
        (tmp_path / f'u{k}.json').write_text(json.dumps(fake_u_so(ctx, ha=cm.sha256(str(sp)), tap=tap)),
                                             encoding='utf-8')
        assert cm.main(['lich'] + base + ['--u-so', str(tmp_path / f'u{k}.json'), '--truoc',
                                          str(tmp_path / prev / 'lich.json'), '--out', str(tmp_path / f'L{k}')]) == 0
        prev = f'L{k}'
        if json.loads((tmp_path / prev / 'lich.json').read_text(encoding='utf-8'))['hoi_tu']:
            break
    L = json.loads((tmp_path / prev / 'lich.json').read_text(encoding='utf-8'))
    assert L['hoi_tu'] and L['chuoi']['xe']['nen_doi'] and L['chuoi']['2']['nen_doi']
    code = cm.main(['track', '--dets', str(d1 / 'dets.npz'), '--session', str(sp), '--run', 'r1', '--u-so',
                    str(tmp_path / f'u{k}.json'), '--lich', str(tmp_path / prev / 'lich.json'),
                    '--out', str(tmp_path / 'tf')])
    assert code == 3
    tr = json.loads((tmp_path / 'tf' / 'track.json').read_text(encoding='utf-8'))
    assert tr['n_hang_kiem_chua_ket_luan'] > 0 and tr['ket_luan'].startswith('chua_ket_luan')
    assert tr['n_chua_ket_luan_xe_hop_le'] > 0
    with open(tmp_path / 'tf' / 'kiem_cheo_khung.csv', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert '1' in {r['chua_ket_luan'] for r in rows if r['chuoi'] == '2'}
    assert '0' in {r['chua_ket_luan'] for r in rows if r['chuoi'] == '2'}      # before the shift: concluded
