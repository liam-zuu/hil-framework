"""c8b_p6_dong (design C8B v7 section 3 (ii); rule c8b_P6_rieng_ii_khoa_v1, CODE 23/11)."""
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import c8b_p6_dong as P  # noqa: E402
import c8_u_imu as cu  # noqa: E402

A = 0.1445
SES = dict(tag_h_m=0.124175, kappa=0.124175 / 2.632)
SES3 = dict(tag_h_m=0.175225, kappa=0.175225 / 2.632)


def _write(path, rows):
    """rows: (idx, valid, scale, speed, wz)."""
    path.write_text('idx,valid,scale,speed,wz\n' + '\n'.join(','.join(map(str, r)) for r in rows) + '\n',
                    encoding='utf-8')


def _run(tmp, name, w_scale, moving):
    """A run: 40 W frames (speed 0) at w_scale, then moving frames [(scale, speed)]."""
    d = tmp / name
    d.mkdir()
    f = d / 'frames.csv'
    rows = [(i, 1, w_scale, 0.0, 0.0) for i in range(40)]
    rows += [(100 + j, 1, s, v, 0.0) for j, (s, v) in enumerate(moving)]
    rows += [(999, 0, 9.9, 0.1, 0.0)]                                          # invalid: ignored
    _write(f, rows)
    tc = dict(schema='tham_chieu_v1', ids=list(range(40)), frames_sha256=cu.sha256(str(f)),
              nguon=[dict(vai_tro='frames', duong_dan=str(f), sha256=cu.sha256(str(f)))])
    p = d / 'tham_chieu_v1.json'
    p.write_text(json.dumps(tc), encoding='utf-8')
    return p, tc


def test_bands_are_the_u3b_of_the_levels():
    b = P.bands_of(A)
    assert [x[0] for x in b] == ['B1', 'B2', 'B3'] and b[0][1] == 0.0 and not b[0][3] and b[1][3]
    assert b[0][2] == pytest.approx(0.3583, abs=1e-4) and b[1][2] == pytest.approx(0.8956, abs=1e-4)
    assert b[2][2] == pytest.approx(1.4330, abs=1e-4)
    assert P.in_band(b[0][2], b[0]) and not P.in_band(b[0][2], b[1])         # closed above, open below for B2


def test_leave_one_out_and_dh(tmp_path):
    """dh = (H - h)(1 - s_W / s); D(X, B) over the OTHER runs, the lock value the max over X (the common max);
    a band without N_MIN frames of another run -> thieu; N added, never subtracted."""
    H, h = 2.632, 0.124175
    p1, t1 = _run(tmp_path, 'r1', 1.0, [(1.01, 0.2)] * 30 + [(0.99, 0.5)] * 30)
    p2, t2 = _run(tmp_path, 'r2', 1.0, [(1.002, 0.2)] * 30)
    f3 = tmp_path / 'b3.csv'
    _write(f3, [(i, 1, 1.0, 0.0, 0.0) for i in range(30)] + [(100 + i, 1, 0.995, 0.3, 0.0) for i in range(30)])
    F3 = P.read_frames(str(f3), None, A)
    out = P.build([('r1', t1), ('r2', t2)], SES, F3, SES3, 0.01, A)
    b1 = out['dai']['B1']
    d_r1 = (H - h) * (1 - 1 / 1.01)
    assert b1['D_theo_luot']['r2'] == pytest.approx(abs(d_r1)) and b1['D_m'] == pytest.approx(abs(d_r1))
    assert b1['D_theo_luot']['r1'] == pytest.approx(abs((H - h) * (1 - 1 / 1.002)))   # r1 left out: only r2
    assert b1['N_m'] == pytest.approx((2.632 - 0.175225) * 0.005) and b1['delta_h_m'] == pytest.approx(
        b1['D_m'] + b1['N_m'])
    b2 = out['dai']['B2']
    assert b2['D_theo_luot']['r1'] is None and b2['D_m'] == pytest.approx(abs((H - h) * (1 - 1 / 0.99)))
    assert b2['trang_thai'] == 'thieu' and b2['N_m'] is None                     # no 3b frame in B2
    assert out['dai']['B0']['N_m'] == 0.0 and out['dai']['B0']['D_m'] == pytest.approx(0.0)
    assert out['dai']['B3']['D_m'] is None and out['trang_thai'] == 'thieu'


def test_refusals(tmp_path):
    p1, t1 = _run(tmp_path, 'r1', 1.0, [])
    with pytest.raises(P.P6Error):
        P.build([('r1', t1)], SES, {}, SES3, 0.01, A)                           # one run: no leave one out
    bad = dict(t1, frames_sha256='ab' * 32)
    with pytest.raises(P.P6Error):
        P.run_dh(bad, A, 2.632, 0.124175)
    with pytest.raises(P.P6Error):
        P.run_dh(dict(t1, ids=[0, 5000]), A, 2.632, 0.124175)                   # a W id not a valid frame


def test_cli(tmp_path):
    p1, _ = _run(tmp_path, 'r1', 1.0, [(1.01, 0.2)] * 30)
    p2, _ = _run(tmp_path, 'r2', 1.0, [(1.002, 0.2)] * 30)
    f3 = tmp_path / 'b3.csv'
    _write(f3, [(i, 1, 1.0, 0.0, 0.0) for i in range(30)] + [(100 + i, 1, 0.995, 0.3, 0.0) for i in range(30)])
    for n, d in (('kq.json', dict(schema=P.KHOA_SCHEMA)), ('ses.json', SES), ('ses3.json', SES3),
                 ('ng.json', dict(format='nguong_khoa_c8_v1', v_d_m_s=dict(gia_tri=0.01)))):
        (tmp_path / n).write_text(json.dumps(d), encoding='utf-8')
    base = ['--khoa-quy-tac', str(tmp_path / 'kq.json'), '--tham-chieu', str(p1), '--tham-chieu', str(p2),
            '--session', str(tmp_path / 'ses.json'), '--b3b-frames', str(f3), '--b3b-session', str(tmp_path / 'ses3.json'),
            '--nguong', str(tmp_path / 'ng.json'), '--canh-ma-m', str(A)]
    assert P.main(base + ['--out', str(tmp_path / 'o.json')]) == 0
    o = json.loads((tmp_path / 'o.json').read_text(encoding='utf-8'))
    assert o['schema'] == P.VERSION and any(n['vai'] == 'khoa_quy_tac' for n in o['nguon'])
    assert P.main(base + ['--out', str(tmp_path / 'o.json')]) == 2                # exists
    (tmp_path / 'kq.json').write_text(json.dumps(dict(schema='x')), encoding='utf-8')
    assert P.main(base + ['--out', str(tmp_path / 'o2.json')]) == 2              # not the lock
