"""Tests of tools/c8b_xoay.py (P2 from turning in place; IMU check). CODE_C8B 18/11/2026."""
import csv
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import c8b_xoay as X  # noqa: E402

FPS = 60.0


def psi_profile(t, w=0.9, still=1.5, turns=2.0):
    """still, +turns, still, -turns, still (heading in rad)."""
    T = turns * 2 * math.pi / w
    a, b, c, d = still, still + T, 2 * still + T, 2 * still + 2 * T
    return np.piecewise(t, [t < a, (t >= a) & (t < b), (t >= b) & (t < c), (t >= c) & (t < d), t >= d],
                        [0.0, lambda u: w * (u - a), w * T, lambda u: w * T - w * (u - c), 0.0]), d + still


def make_place(tmp_path, name, q=(0.004, -0.0025), c0=(1.0, 0.0), v=(0.0, 0.0), noise=0.0003, psi0=0.3,
               cam_err_deg=0.0, seed=1, skip_dir=None, w=0.9):
    rng = np.random.default_rng(seed)
    _, t_end = psi_profile(np.array([0.0]), w=w)
    t = np.arange(0.0, t_end, 1 / FPS)
    psi, _ = psi_profile(t, w=w)
    psi = psi + psi0
    if skip_dir == -1:
        keep = t < (t_end / 2)
        t, psi = t[keep], psi[keep]
    c = np.array(c0)[None, :] + (t - t.mean())[:, None] * np.array(v)[None, :]
    R = np.stack([np.cos(psi), -np.sin(psi), np.sin(psi), np.cos(psi)], axis=1).reshape(-1, 2, 2)
    tag = c + R @ np.array(q)
    tag = tag + rng.normal(0, noise, tag.shape)
    psi_cam = psi + np.radians(cam_err_deg) * np.sin(psi) + rng.normal(0, np.radians(0.03), psi.shape)
    p = tmp_path / f'frames_{name}.csv'
    with open(p, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['t_s', 'valid', 'car_seen', 'tag_x', 'tag_y', 'tag_psi'])
        for k in range(len(t)):
            a = (psi_cam[k] + math.pi) % (2 * math.pi) - math.pi
            wr.writerow([f'{t[k]:.6f}', 1, 1, f'{tag[k, 0]:.6f}', f'{tag[k, 1]:.6f}', f'{a:.7f}'])
    return str(p), t, psi


def make_log(tmp_path, name, t_cam, psi_true, off=1234.5, scale=1.003):
    t = np.arange(t_cam[0] - 2.0, t_cam[-1] + 2.0, 0.05)
    psi = np.interp(t, t_cam, psi_true) * scale
    gz = np.gradient(psi, 0.05)
    p = tmp_path / f'log_{name}.csv'
    with open(p, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['timestamp_us', 'yaw_rad', 'rv_n', 'age_r_ms', 'gz', 'gyro_n', 'age_g_ms', 'reset_n'])
        for k in range(len(t)):
            y = (psi[k] + math.pi) % (2 * math.pi) - math.pi
            wr.writerow([int(round((t[k] + off) * 1e6)), f'{y:.7f}', k + 1, 0, f'{gz[k]:.6f}', k + 1, 0, 0])
    return str(p)


def test_offset_recovered_two_places(tmp_path):
    q = (0.004, -0.0025)
    pO, *_ = make_place(tmp_path, 'O', q=q, c0=(0.0, 0.0), seed=1)
    pK, *_ = make_place(tmp_path, 'K1', q=q, c0=(1.0, 0.0), seed=2)
    d = X.analyse({'O': pO, 'K1': pK})
    assert d['version'] == 'c8b_xoay_v1' and d['n_doan'] == 4
    assert np.hypot(d['q_tag_m'][0] - q[0], d['q_tag_m'][1] - q[1]) < 1e-4
    assert d['du_m'] == pytest.approx(d['spread_m'] + d['se3_m'])
    assert d['du_m'] >= max(np.hypot(e['q_m'][0] - d['q_tag_m'][0], e['q_m'][1] - d['q_tag_m'][1])
                            for c in d['cho'].values() for e in c['doan'])
    assert {e['chieu'] for e in d['cho']['O']['doan']} == {1, -1}
    assert d['troi_lon'] == []


def test_drift_separated_and_flagged(tmp_path):
    q = (0.003, 0.002)
    pO, *_ = make_place(tmp_path, 'O', q=q, v=(0.002, -0.001), seed=3)       # creeping centre, 2 mm/s
    pK, *_ = make_place(tmp_path, 'K1', q=q, c0=(1.0, 0.0), seed=4)
    d = X.analyse({'O': pO, 'K1': pK})
    assert np.hypot(d['q_tag_m'][0] - q[0], d['q_tag_m'][1] - q[1]) < 1.5e-4   # drift does not bias q
    assert any(s.startswith('O:') for s in d['troi_lon'])
    v = d['cho']['O']['doan'][0]['v_m_s']
    assert v == pytest.approx([0.002, -0.001], abs=2e-4)


def test_different_offsets_widen_du(tmp_path):
    # a place where the turning centre is off (slip about another point) shows as spread, not hidden
    pO, *_ = make_place(tmp_path, 'O', q=(0.004, 0.0), seed=5)
    pK, *_ = make_place(tmp_path, 'K1', q=(0.000, 0.0), c0=(1.0, 0.0), seed=6)
    d = X.analyse({'O': pO, 'K1': pK})
    assert d['spread_m'] == pytest.approx(0.002, abs=2e-4) and d['du_m'] >= 0.002


def test_refusals(tmp_path):
    pO, *_ = make_place(tmp_path, 'O', seed=7)
    with pytest.raises(X.XError, match='two places'):
        X.analyse({'O': pO})
    pK, *_ = make_place(tmp_path, 'K1', skip_dir=-1, seed=8)
    with pytest.raises(X.XError, match='each direction'):
        X.analyse({'O': pO, 'K1': pK})
    p = tmp_path / 'bad.csv'
    p.write_text('t_s,valid,car_seen,tag_x,tag_y\n0,1,1,0,0\n', encoding='utf-8')
    with pytest.raises(X.XError, match='no column'):
        X.analyse({'O': pO, 'K1': str(p)})


def test_short_turn_not_an_interval(tmp_path):
    pO, *_ = make_place(tmp_path, 'O', seed=9)
    F = X.read_frames(pO)
    segs, short = X.intervals(F, min_turn=5 * math.pi)        # 2 turns < 2.5 turns: both dropped as short
    assert segs == [] and len(short) == 2


def test_imu_check_offset_and_heading_error(tmp_path):
    pO, tO, psiO = make_place(tmp_path, 'O', cam_err_deg=0.1, seed=10)
    pK, tK, psiK = make_place(tmp_path, 'K1', c0=(1.0, 0.0), seed=11)
    lO = make_log(tmp_path, 'O', tO, psiO, off=1234.5)
    d = X.analyse({'O': pO, 'K1': pK}, {'O': lO})
    imu = d['cho']['O']['imu']
    assert imu['offset_s'] == pytest.approx(1234.5, abs=0.01)
    for s in imu['doan']:
        assert s['ty_le'] == pytest.approx(-0.003, abs=5e-4)                  # camera - IMU slope: IMU scale removed
        assert 0.07 < s['du_max_deg'] < 0.25                                 # the 0.1 deg heading-dependent error
    assert 'imu' not in d['cho']['K1']


def test_cli_writes_once(tmp_path):
    pO, *_ = make_place(tmp_path, 'O', seed=12)
    pK, *_ = make_place(tmp_path, 'K1', c0=(1.0, 0.0), seed=13)
    out = tmp_path / 'xoay.json'
    assert X.main(['--cho', f'O={pO}', '--cho', f'K1={pK}', '--out', str(out)]) == 0
    d = json.loads(out.read_text(encoding='utf-8'))
    assert d['version'] == 'c8b_xoay_v1' and 'script' in d
    assert X.main(['--cho', f'O={pO}', '--cho', f'K1={pK}', '--out', str(out)]) == 2
    assert X.main(['--cho', f'O={pO}', '--log', f'K9={pO}', '--out', str(tmp_path / 'b.json')]) == 2
