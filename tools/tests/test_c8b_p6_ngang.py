"""c8b_p6_ngang (rule c8b_P6_rieng_ii_khoa_v2: lateral width of the tag against the other runs at the same pose)."""
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import c8b_p6_ngang as P  # noqa: E402
import c8b_a6 as a6m  # noqa: E402
import c8_u_imu as cu  # noqa: E402

A = 0.1445
SES = dict(tag_h_m=0.124175, kappa=0.124175 / 2.632)
SES3 = dict(tag_h_m=0.175225, kappa=0.175225 / 2.632)
NG = dict(format='nguong_khoa_c8_v1', v_d_m_s=dict(gia_tri=0.01), quy_uoc_khac=dict(pred_m=0.02, yaw_nghi_deg=3.0))
K = 2.632 - 0.124175


def corners(x, y, psi, sx=1.0, sy=1.0):
    """Corners of the tag at (x, y), heading psi, stretched sx along the body x axis and sy along the body y axis."""
    r = a6m.body(A) * np.array([sx, sy])
    c, s = math.cos(psi), math.sin(psi)
    return (r @ np.array([[c, s], [-s, c]]) + np.array([x, y]))


def test_widths_separate_the_two_axes():
    P4 = np.array([corners(0.3, -0.2, 0.7, sx=0.97, sy=1.004)])
    lx, ly, T, psi = P.widths(P4, A)
    assert lx[0] == pytest.approx(0.97) and ly[0] == pytest.approx(1.004)
    assert T[0] == pytest.approx([0.3, -0.2]) and psi[0] == pytest.approx(0.7)


def _run(tmp, name, frames, session='s1'):
    """frames: (x, y, psi, sx, sy, speed)."""
    d = tmp / name
    d.mkdir()
    f = d / 'frames.csv'
    f.write_text('idx,valid,scale,speed,wz\n' + '\n'.join(f'{i},1,1.0,{fr[5]},0.0' for i, fr in enumerate(frames))
                 + f'\n{len(frames)},0,1.0,0.3,0.0\n', encoding='utf-8')
    g = d / 'goc.json'
    g.write_text(json.dumps(dict(schema='c8b_a6_goc_v1', luot_id=name, session_sha256=session,
                                 frames_sha256=cu.sha256(str(f)),
                                 khung=[[i, i / 60, corners(*fr[:5]).ravel().tolist()] for i, fr in enumerate(frames)]
                                 + [[len(frames), 0.0, corners(0, 0, 0).ravel().tolist()]])), encoding='utf-8')
    return f'{g}={f}', P.read_run(str(g), str(f), A)


def test_blur_along_travel_is_not_height_but_height_is(tmp_path):
    """Same pose in two runs: a shrink along the travel axis (blur) gives 0; an equal change of both axes (height)
    gives (H - h) x that change; another pose or another camera session is not compared."""
    still = [(0.0, 0.0, 0.5, 1.0, 1.0, 0.0)] * 40
    mov = [(0.5, 0.1, 0.9, 1.0, 1.0, 0.3)] * 40
    _, r1 = _run(tmp_path, 'r1', still + mov)
    _, r2 = _run(tmp_path, 'r2', still + [(0.5, 0.1, 0.9, 0.97, 1.0, 0.3)] * 40)          # blur: x only
    _, r3 = _run(tmp_path, 'r3', still + [(0.5, 0.1, 0.9, 1.002, 1.002, 0.3)] * 40)       # height: both axes
    o = P.build([r1, r2], SES, NG, A)
    assert o['dai']['B1']['D_m'] == pytest.approx(0.0, abs=1e-12) and o['dai']['B0']['D_m'] == pytest.approx(0.0, abs=1e-12)
    assert o['dai']['B1']['n_do'] == 160 and o['dai']['B0']['n_do'] == 80
    o = P.build([r1, r3], SES, NG, A)
    assert o['dai']['B1']['D_m'] == pytest.approx(K * 0.002, rel=1e-3)
    assert o['dai']['B1']['trang_thai'] == 'thieu' and o['dai']['B1']['delta_h_m'] is None   # no jig N: not complete
    assert o['dai']['B0']['delta_h_m'] == pytest.approx(0.0, abs=1e-12) and o['dai']['B0']['trang_thai'] == 'dat'
    _, far = _run(tmp_path, 'far', still + [(0.53, 0.1, 0.9, 1.002, 1.002, 0.3)] * 40)    # 3 cm away
    _, turn = _run(tmp_path, 'turn', still + [(0.5, 0.1, 0.9 + math.radians(4), 1.002, 1.002, 0.3)] * 40)
    _, ses2 = _run(tmp_path, 'ses2', still + [(0.5, 0.1, 0.9, 1.002, 1.002, 0.3)] * 40, session='s2')
    for other in (far, turn, ses2):
        o = P.build([r1, other], SES, NG, A)
        assert o['dai']['B1']['n_do'] == (0 if other is ses2 else 80)                        # only the still frames
        assert o['dai']['B1']['D_m'] in (None, pytest.approx(0.0, abs=1e-12))
    with pytest.raises(P.P6Error):
        P.build([r1], SES, NG, A)


def test_reference_needs_three_frames_and_band_thirty(tmp_path):
    still = [(0.0, 0.0, 0.5, 1.0, 1.0, 0.0)] * 40
    _, r1 = _run(tmp_path, 'r1', still + [(0.5, 0.1, 0.9, 1.0, 1.01, 0.5)] * 20)
    _, r2 = _run(tmp_path, 'r2', still + [(0.5, 0.1, 0.9, 1.0, 1.0, 0.5)] * 2)              # 2 < N_REF neighbours
    o = P.build([r1, r2], SES, NG, A)
    assert o['dai']['B2']['n_khung'] == 22 and o['dai']['B2']['n_do'] == 2 and o['dai']['B2']['D_m'] is None
    assert o['trang_thai'] == 'thieu'


def _ga(tmp, n_slow=40, n_fast=40, shrink=0.99, axis='x'):
    """A jig: the tag at one pose, travel along the tag x axis in the image; fast frames shrink along `axis`."""
    fr = tmp / 'ga_frames.csv'
    rows, px = [], []
    for i in range(n_slow + n_fast):
        fast = i >= n_slow
        rows.append(f'{i},1,1.0,{0.3 if fast else 0.0},0.0,0.2,0.1,0.0')
        sx, sy = (shrink, 1.0) if axis == 'x' else (1.0, shrink)
        c = corners(100.0 + 3.0 * i, 50.0, 0.0, *( (sx, sy) if fast else (1.0, 1.0))) * 1.0
        px.append(c * 1.0)
    fr.write_text('idx,valid,scale,speed,wz,x,y,psi\n' + '\n'.join(rows) + '\n', encoding='utf-8')
    z = tmp / 'ga.npz'
    np.savez(z, schema=P.GA_SCHEMA, video='v.MOV', video_sha256='ab' * 32, n_frames=len(rows),
             k=np.arange(len(rows)), t=np.arange(len(rows)) / 60.0, px=np.array(px))
    return z, fr


def test_jig_takes_the_axis_across_travel(tmp_path):
    """Jig B (item 6): travel along the tag x axis -> the y width is used; a shrink along x (blur) gives N 0, a shrink
    along y gives (H3 - h3) x it; N(B0) = 0."""
    z, fr = _ga(tmp_path, shrink=0.99, axis='x')
    rows, src = P.read_ga([(str(z), 0)], str(fr), A)
    gab, info = P.do_ga(rows, A, 2.632, 0.175225, 0.01, 0.02, math.radians(3))
    assert info['truc_ngang'] == 'y' and info['goc_huong_chay_truc_x_ma_deg'] == pytest.approx(0.0, abs=1e-6)
    assert gab['B1']['N_m'] == pytest.approx(0.0, abs=1e-9) and gab['B1']['n_do_ga'] == 40
    (tmp_path / 'b').mkdir()
    z, fr = _ga(tmp_path / 'b', shrink=0.99, axis='y')
    rows, _ = P.read_ga([(str(z), 0)], str(fr), A)
    gab, _ = P.do_ga(rows, A, 2.632, 0.175225, 0.01, 0.02, math.radians(3))
    assert gab['B1']['N_m'] == pytest.approx((2.632 - 0.175225) * 0.01, rel=1e-6) and gab['B2']['N_m'] is None


def test_cli(tmp_path):
    still = [(0.0, 0.0, 0.5, 1.0, 1.0, 0.0)] * 40
    a1, _ = _run(tmp_path, 'r1', still + [(0.5, 0.1, 0.9, 1.0, 1.0, 0.3)] * 40)
    a2, _ = _run(tmp_path, 'r2', still + [(0.5, 0.1, 0.9, 0.98, 1.001, 0.3)] * 40)
    (tmp_path / 'g').mkdir()
    z, fr = _ga(tmp_path / 'g')
    for n, d in (('kq.json', dict(schema=P.KHOA_SCHEMA)), ('ses.json', SES), ('ses3.json', SES3), ('ng.json', NG)):
        (tmp_path / n).write_text(json.dumps(d), encoding='utf-8')
    base = ['tinh', '--khoa-quy-tac', str(tmp_path / 'kq.json'), '--luot', a1, '--luot', a2, '--session',
            str(tmp_path / 'ses.json'), '--nguong', str(tmp_path / 'ng.json'), '--canh-ma-m', str(A)]
    ga = ['--ga-goc', f'{z}=0', '--ga-frames', str(fr), '--ga-session', str(tmp_path / 'ses3.json')]
    assert P.main(base + ga + ['--out', str(tmp_path / 'o.json')]) == 0
    o = json.loads((tmp_path / 'o.json').read_text(encoding='utf-8'))
    assert o['schema'] == P.VERSION and o['dai']['B1']['delta_h_m'] == pytest.approx(K * 0.001, rel=1e-3)
    assert o['dai']['B1']['trang_thai'] == 'dat' and o['ga_B']['truc_ngang'] == 'y'
    assert any(n['vai'] == 'ga_goc' for n in o['nguon']) and any(n['vai'] == 'khoa_quy_tac' for n in o['nguon'])
    assert P.main(base + ga + ['--out', str(tmp_path / 'o.json')]) == 2                       # exists
    assert P.main(base + ga[:2] + ['--out', str(tmp_path / 'x.json')]) == 2                   # jig options together
    (tmp_path / 'kq.json').write_text(json.dumps(dict(schema='c8b_P6_rieng_ii_khoa_v1')), encoding='utf-8')
    assert P.main(base + ['--out', str(tmp_path / 'y.json')]) == 2                            # the v1 lock refused


def test_delta_is_d_plus_n(tmp_path):
    """Lock item 7: delta_h_dong = D + N (added, never subtracted): a height change of the runs and a residual blur
    of the jig along its lateral axis both count."""
    still = [(0.0, 0.0, 0.5, 1.0, 1.0, 0.0)] * 40
    _, r1 = _run(tmp_path, 'r1', still + [(0.5, 0.1, 0.9, 1.0, 1.0, 0.3)] * 40)
    _, r2 = _run(tmp_path, 'r2', still + [(0.5, 0.1, 0.9, 1.0, 1.002, 0.3)] * 40)
    (tmp_path / 'g').mkdir()
    z, fr = _ga(tmp_path / 'g', shrink=0.999, axis='y')
    rows, _ = P.read_ga([(str(z), 0)], str(fr), A)
    o = P.build([r1, r2], SES, NG, A, rows, SES3)
    d = o['dai']['B1']
    assert d['D_m'] == pytest.approx(K * 0.002, rel=1e-3) and d['N_m'] == pytest.approx((2.632 - 0.175225) * 0.001, rel=1e-3)
    assert d['delta_h_m'] == pytest.approx(d['D_m'] + d['N_m'], rel=1e-12) and d['trang_thai'] == 'dat'
    assert o['dai']['B0']['N_m'] == 0.0 and o['dai']['B2']['trang_thai'] == 'thieu'
