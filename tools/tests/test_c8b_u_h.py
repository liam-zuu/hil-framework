"""Tests of tools/c8b_u_h.py (u_h_chung from three 3b jigs and the roof tag; CODE_C8B 22/11)."""
import csv
import hashlib
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import c8b_u_h as uh  # noqa: E402
import c8b_a6 as a6  # noqa: E402

H = 2.632
K_TRUE = 0.987


def sha(p):
    with open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def session(path, h_a):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(dict(kappa=h_a / H, tag_h_m=h_a, car_side_m=0.1445), f)
    return path


def jig_frames(path, h_a, h_t, k=K_TRUE, fast_bias=0.0, n=3000, noise=0.0, seed=0):
    """A spinning jig slowing down from 6.4 to 0.4 rad/s; scale = k (H - h_a) / (H - h_t), plus fast_bias on the
    frames faster than 2 rad/s (blur), plus noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 60.0
    om = 6.0 * (1 - t / t[-1]) + 0.4
    psi = np.concatenate([[0.0], np.cumsum(0.5 * (om[1:] + om[:-1]) * np.diff(t))])
    s0 = k * (H - h_a) / (H - h_t)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['idx', 't_s', 'valid', 'x', 'y', 'psi', 'tag_x', 'tag_y', 'scale', 'row_px', 'col_px'])
        for i in range(n):
            sc = s0 + (fast_bias if om[i] > 2.0 else 0.0) + noise * rng.standard_normal()
            w.writerow([i, f'{t[i]:.6f}', 1, 0.5, 0.3, f'{math.atan2(math.sin(psi[i]), math.cos(psi[i])):.9f}',
                        0.5, 0.3, f'{sc:.10f}', 1000.0, 1000.0])
    return path


def track(dirp, h_a, h_t, ses_sha, video_sha='v' * 64, k=K_TRUE, n=400, bad_edges=True, car_off=()):
    os.makedirs(dirp, exist_ok=True)
    s0 = k * (H - h_a) / (H - h_t)
    fp = os.path.join(dirp, 'frames.csv')
    with open(fp, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['idx', 't_s', 'valid', 'car_seen', 'scale'])
        for i in range(n):
            t = i / 30.0
            sc = s0
            if bad_edges and (t < 0.5 or t > (n - 1) / 30.0 - 0.5):
                sc = 0.5                                   # the trimmed ends must not count
            w.writerow([i, f'{t:.6f}', 1, 0 if i in car_off else 1, f'{sc:.10f}'])
    with open(os.path.join(dirp, 'track.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(video_sha256=video_sha, session_sha256=ses_sha, frames_sha256=sha(fp)), f)
    return dirp


@pytest.fixture
def setup(tmp_path):
    def make(h_true=(0.13425, 0.2, 0.175225), h_noc_true=0.130, fast_bias=0.0, n_track=400, car_off=(),
             static=True):
        g = tmp_path
        cal = [0.13425, 0.2, 0.175225]
        ga = []
        for name, hc, ht in zip(('A', 'A2', 'B'), cal, h_true):
            sp = session(str(g / f'ses_{name}.json'), hc)
            fp = jig_frames(str(g / f'fr_{name}.csv'), hc, ht, fast_bias=fast_bias)
            ga.append(dict(ten=name, h_cal_m=hc, frames=os.path.basename(fp), frames_sha256=sha(fp),
                           session=os.path.basename(sp), session_sha256=sha(sp)))
        sn = session(str(g / 'ses_H0a.json'), 0.119)
        track(str(g / 'track_N0'), 0.119, h_noc_true, sha(sn), n=n_track, car_off=car_off)
        K = dict(schema=uh.SCHEMA_KHOA, tam_id='tam_c8_1', h_m=0.124175, ga=ga,
                 noc=dict(h_cal_m=0.124175, video_sha256='v' * 64, track='track_N0', trim_s=0.5, n_min=100,
                          session='ses_H0a.json', session_sha256=sha(sn)))
        if static:
            track(str(g / 'tinh_A'), 0.119, 0.13425, sha(sn), n=150, bad_edges=False)
            K['kiem_cheo_tinh_A'] = dict(h_cal_m=0.13425, frames='tinh_A/frames.csv',
                                         frames_sha256=sha(str(g / 'tinh_A' / 'frames.csv')),
                                         session='ses_H0a.json', session_sha256=sha(sn))
        kp = str(g / 'khoa.json')
        with open(kp, 'w', encoding='utf-8') as f:
            json.dump(K, f)
        sc = session(str(g / 'ses_cham.json'), 0.124175)
        return kp, str(g), [sc], str(g / 'out.json')
    return make


def test_inverse():
    assert abs(uh.k_of(0.99, H, 0.119, 0.13) - 0.99 * (H - 0.13) / (H - 0.119)) < 1e-15
    k = uh.k_of(0.98, H, 0.119, 0.131)
    assert abs(uh.h_of(0.98, k, H, 0.119) - 0.131) < 1e-12


def test_exact_recovery(setup):
    kp, goc, sc, out = setup()
    d = uh.run(kp, goc, sc, out)
    assert d['trang_thai'] == 'dat'
    assert d['phan']['do_trai_ba_ga_m'] < 1e-6
    assert abs(d['h_video_m'] - 0.130) < 1e-6
    assert abs(d['phan']['lech_video_so_tay_m'] - (0.130 - 0.124175)) < 1e-6
    assert abs(d['u_h_chung_m'] - (d['phan']['do_trai_ba_ga_m'] + d['phan']['lech_video_so_tay_m'] + 1e-4)) < 1e-12
    assert d['n_ga_giu_ngoai'] == 3 and d['h_m'] == 0.124175
    v, w = a6.kiem_u_h(json.load(open(out, encoding='utf-8')), 'tam_c8_1', 0.124175)
    assert v is not None and not w
    ka = d['kiem_cheo']['tinh_A']
    assert abs(ka['h_du_bao_k_giu_ngoai_m'] - 0.13425) < 1e-6


def test_held_out_value(setup):
    """B 3 mm higher than its caliper number: k_B is off, the held-out errors follow the closed form."""
    kp, goc, sc, out = setup(h_true=(0.13425, 0.2, 0.178225))
    d = uh.run(kp, goc, sc, out)
    cal = [0.13425, 0.2, 0.175225]
    tru = [0.13425, 0.2, 0.178225]
    s = [K_TRUE * (H - c) / (H - t) for c, t in zip(cal, tru)]
    ks = [s[j] for j in range(3)]                                # h_a = h_cal, so k_j = s_j
    errs = []
    for j in range(3):
        ko = np.mean([ks[i] for i in range(3) if i != j])
        errs.append(abs(H - ko * (H - cal[j]) / s[j] - cal[j]))
    assert abs(d['phan']['do_trai_ba_ga_m'] - max(errs)) < 1e-6
    assert d['phan']['do_trai_ba_ga_m'] > 0.002
    kc = np.mean(ks)
    sn = K_TRUE * (H - 0.119) / (H - 0.130)
    assert abs(d['h_video_m'] - (H - kc * (H - 0.119) / sn)) < 1e-6


def test_slowest_frames_only(setup):
    """A blur bias on the fast frames must not reach s_j (c8b_3b.slowest)."""
    kp, goc, sc, out = setup(fast_bias=0.01)
    d = uh.run(kp, goc, sc, out)
    assert d['trang_thai'] == 'dat'
    assert d['phan']['do_trai_ba_ga_m'] < 1e-6
    assert abs(d['ga'][0]['s'] - K_TRUE) < 1e-9


def test_jig_sha_refused(setup):
    kp, goc, sc, out = setup()
    with open(os.path.join(goc, 'fr_A2.csv'), 'a', encoding='utf-8') as f:
        f.write('\n')
    d = uh.run(kp, goc, sc, out)
    assert d['trang_thai'] == 'thieu' and any('A2' in w and 'sha256' in w for w in d['ly_do'])


def test_session_sha_refused(setup):
    kp, goc, sc, out = setup()
    with open(os.path.join(goc, 'ses_B.json'), 'a', encoding='utf-8') as f:
        f.write(' ')
    with pytest.raises(uh.UhError):
        uh.run(kp, goc, sc, out)


def test_session_cham_h(setup):
    kp, goc, sc, out = setup()
    bad = session(os.path.join(goc, 'ses_bad.json'), 0.119)
    with pytest.raises(uh.UhError):
        uh.run(kp, goc, [bad], out)
    with pytest.raises(uh.UhError):
        uh.run(kp, goc, [], out)


def test_roof_window(setup):
    """Trimmed ends (scale 0.5) and frames without the car do not count; too few frames -> thieu."""
    kp, goc, sc, out = setup(car_off=tuple(range(20, 60)))
    d = uh.run(kp, goc, sc, out)
    assert d['trang_thai'] == 'dat' and abs(d['h_video_m'] - 0.130) < 1e-6
    assert d['noc']['n'] == 400 - 15 - 15 - 40
    kp, goc, sc, out = setup(n_track=120)
    d = uh.run(kp, goc, sc, out)
    assert d['trang_thai'] == 'thieu' and any('still frames' in w for w in d['ly_do'])


def test_video_sha_of_track(setup):
    kp, goc, sc, out = setup()
    tj = os.path.join(goc, 'track_N0', 'track.json')
    t = json.load(open(tj, encoding='utf-8'))
    t['video_sha256'] = 'w' * 64
    json.dump(t, open(tj, 'w', encoding='utf-8'))
    d = uh.run(kp, goc, sc, out)
    assert d['trang_thai'] == 'thieu' and any('video_sha256' in w for w in d['ly_do'])


def test_frames_sha_of_track(setup):
    kp, goc, sc, out = setup()
    fp = os.path.join(goc, 'track_N0', 'frames.csv')
    rows = open(fp, encoding='utf-8').read().splitlines()
    rows[100] = rows[100] + '0'                                  # same scale value, other bytes
    open(fp, 'w', encoding='utf-8').write('\n'.join(rows) + '\n')
    d = uh.run(kp, goc, sc, out)
    assert d['trang_thai'] == 'thieu' and any('frames_sha256' in w for w in d['ly_do'])


def test_few_spinning_frames(setup, tmp_path):
    kp, goc, sc, out = setup()
    fp = os.path.join(goc, 'fr_B.csv')
    rows = open(fp, encoding='utf-8').read().splitlines()
    open(fp, 'w', encoding='utf-8').write('\n'.join(rows[:41]) + '\n')     # 40 frames < 50
    K = json.load(open(kp, encoding='utf-8'))
    K['ga'][2]['frames_sha256'] = sha(fp)
    json.dump(K, open(kp, 'w', encoding='utf-8'))
    d = uh.run(kp, goc, sc, out)
    assert d['trang_thai'] == 'thieu' and any(w.startswith('ga B') for w in d['ly_do'])
    assert d['u_h_chung_m'] is None


def test_lock_checks(setup):
    kp, goc, sc, out = setup()
    K = json.load(open(kp, encoding='utf-8'))
    K['ga'] = K['ga'][:2]
    json.dump(K, open(kp, 'w', encoding='utf-8'))
    with pytest.raises(uh.UhError):
        uh.run(kp, goc, sc, out)
    K = json.load(open(kp, encoding='utf-8'))
    K['schema'] = 'x'
    json.dump(K, open(kp, 'w', encoding='utf-8'))
    with pytest.raises(uh.UhError):
        uh.load_khoa(kp)


def test_cli(setup, capsys):
    kp, goc, sc, out = setup()
    rc = uh.main(['--khoa', kp, '--goc', goc, '--session-cham', *sc, '--out', out])
    assert rc == 0 and capsys.readouterr().out.strip().splitlines()[-1] == '#ket_qua,ma=0,trang_thai=dat'
    rc = uh.main(['--khoa', kp, '--goc', goc, '--out', out])
    assert rc == 2
