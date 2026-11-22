"""F4 xoay_v1 per place of c8b_xoay (contract F4, CODE_C8B 22/11): E window from F2, psi' after A6, readers."""
import csv
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c8b_a6 as A6  # noqa: E402
import c8b_xoay as X  # noqa: E402
from test_c8b_xoay import make_place  # noqa: E402

PLUMB = list(A6.PLUMB_L125)
ZC = A6.ZC_L125
HB = 0.124175
STILL = 1.5


def fake_a6(g=(0.0, 0.0), dat=True, ses=('s1',), tam='T1'):
    return dict(schema=A6.SCHEMA, tam_id=tam, trang_thai=dict(a6='dat' if dat else 'thieu', h='thieu'),
                g_b=list(g), eps_g=0.0005, e_bc_rad=1e-5, plumb_xy_m=PLUMB, H_m=ZC, h_m=HB, side_m=0.1445,
                mien=dict(rho_max_m=3.0), session=[dict(session_sha256=s) for s in ses], _sha256='a' * 64)


def tilt_frames(path, g):
    """Add the A6 heading error dpsi_model(g) to the psi column (the camera heading of a tilted tag) and refresh the
    frames_sha256 of track.json."""
    import c7_measure as CM
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        t, p = np.array([float(r['tag_x']), float(r['tag_y'])]), float(r['psi'])
        r['psi'] = repr(p + A6.dpsi_model(g, t, p, PLUMB, ZC, HB))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    tj = os.path.join(os.path.dirname(path), 'track.json')
    info = json.load(open(tj, encoding='utf-8'))
    info['frames_sha256'] = CM.sha256(path)
    json.dump(info, open(tj, 'w', encoding='utf-8'))


def f2_rows(path, t, cho='goc', ses='s1', twice=None, sha=None):
    import c7_measure as CM
    s = sha or CM.sha256(path)
    n = len(t)
    a = int(np.searchsorted(t, STILL - 0.2))
    b = int(np.searchsorted(t, t[-1] - STILL + 0.2))
    mk = lambda loai, ids: dict(schema='cua_so_v1', cho=cho, loai=loai, luot_id=f'HC_{cho}', clip=f'{cho}.mov',
                                clip_sha256=None, session_sha256=ses, frames_sha256=s, chi_so=0,
                                khung=[dict(id=int(i), pts_s=float(t[i])) for i in ids])
    rows = [mk('E_dau', range(0, a)), mk('E_cuoi', range(b, n))]
    if twice:
        rows.append(mk(twice, range(0, 5)))
    return rows


def write_rows(tmp_path, rows, name='cs.jsonl'):
    p = tmp_path / name
    p.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    return str(p)


def place(tmp_path, name, q, c0, g=None, seed=1):
    p, t, _ = make_place(tmp_path, name, q=q, c0=c0, noise=0.0001, seed=seed)
    if g is not None:
        tilt_frames(p, g)
    return p, t


def test_f4_q_after_a6_and_fields(tmp_path):
    g = (0.05, -0.03)                                   # large tilt: the A6 heading error biases q when not removed
    q = (0.004, -0.0025)
    pg, tg = place(tmp_path, 'goc', q, (1.0, 0.0), g)
    pk, tk = place(tmp_path, 'K1', q, (0.2, 1.1), g, seed=2)
    rows = f2_rows(pg, tg, 'goc') + f2_rows(pk, tk, 'K1')
    d = X.theo_cho(dict(goc=pg, K1=pk), fake_a6(g), rows, 'T1')
    d0 = X.theo_cho(dict(goc=pg, K1=pk), fake_a6((0.0, 0.0)), rows, 'T1')
    assert d['version'] == d['schema'] == 'xoay_v1' and d['a6_sha256'] == 'a' * 64
    for k in ('goc', 'K1'):
        b, b0 = d['cho'][k], d0['cho'][k]
        assert b['trang_thai'] == 'dat', b['ly_do']
        err = math.hypot(b['q_tag_m'][0] - q[0], b['q_tag_m'][1] - q[1])
        assert err <= b['u_q_m'] and b0['trang_thai'] == 'dat', (k, err, b['u_q_m'])
        assert b['du_m'] == b['u_q_m'] and b['d_vet_m'] == pytest.approx(math.hypot(*b['q_tag_m']) + b['u_q_m'])
        assert b['d_vet_dat'] is True and b['u_r_m'] > 0 and len(b['r_floor_m']) == 2
        assert set(b['hai_chieu']) == {'+', '-', 'lech_m'} and b['cua_so_E']['E_doan'][0] > 0
        assert b['n_khung_a6_khong_dat'] == 0
    assert d['cho']['goc']['r_floor_m'] == pytest.approx([1.0, 0.0], abs=5e-4)
    assert d['cho']['K1']['r_floor_m'] == pytest.approx([0.2, 1.1], abs=5e-4)


def test_f4_psi_prime_is_psi_minus_a6(tmp_path, monkeypatch):
    # a constant A6 correction delta: the fit on psi' = psi - delta gives R(delta) q (sign of the subtraction, F3)
    q, delta = (0.004, -0.0025), 0.3
    pg, tg = place(tmp_path, 'goc', q, (1.0, 0.0))
    monkeypatch.setattr(A6, 'delta_psi_a6', lambda a6, xy, psi, C, H: (np.full(len(psi), delta),
                                                                       np.ones(len(psi), bool), [''] * len(psi)))
    b = X.theo_cho(dict(goc=pg), fake_a6(), f2_rows(pg, tg), 'T1')['cho']['goc']
    c, s_ = math.cos(delta), math.sin(delta)
    assert b['trang_thai'] == 'dat'
    assert b['q_tag_m'] == pytest.approx([c * q[0] - s_ * q[1], s_ * q[0] + c * q[1]], abs=3e-5)


def test_f4_only_e_doan_frames_used(tmp_path):
    pg, tg = place(tmp_path, 'goc', (0.004, -0.0025), (1.0, 0.0))
    rows = f2_rows(pg, tg)
    b = X.theo_cho(dict(goc=pg), fake_a6(), rows, 'T1')['cho']['goc']
    e = b['cua_so_E']['E_doan']
    assert b['n_khung_E'] == e[1] - e[0] + 1 and b['n_khung_E'] < len(tg)


@pytest.mark.parametrize('case', ['a6_not_dat', 'other_plate', 'no_e_cuoi', 'two_e_dau', 'other_frames', 'session'])
def test_f4_undecided_not_guessed(tmp_path, case):
    pg, tg = place(tmp_path, 'goc', (0.004, -0.0025), (1.0, 0.0))
    rows = f2_rows(pg, tg, twice='E_dau' if case == 'two_e_dau' else None,
                   sha='b' * 64 if case == 'other_frames' else None)
    if case == 'no_e_cuoi':
        rows = [r for r in rows if r['loai'] != 'E_cuoi']
    a6 = fake_a6(dat=case != 'a6_not_dat', tam='T2' if case == 'other_plate' else 'T1',
                 ses=('s9',) if case == 'session' else ('s1',))
    b = X.theo_cho(dict(goc=pg), a6, rows, 'T1')['cho']['goc']
    assert b['trang_thai'] == 'chua_ket_luan' and b['ly_do'], case
    if case in ('a6_not_dat', 'other_plate', 'no_e_cuoi', 'two_e_dau', 'other_frames'):
        assert b['q_tag_m'] is None and b['u_q_m'] is None, case
    word = dict(a6_not_dat='A6', other_plate='plate', no_e_cuoi='E_cuoi', two_e_dau='E_dau', other_frames='E_dau',
                session='session')[case]
    assert any(word in w for w in b['ly_do']), (case, b['ly_do'])


def test_f4_track_json_clip_and_session_checked(tmp_path):
    pg, tg = place(tmp_path, 'goc', (0.004, -0.0025), (1.0, 0.0))
    tj = os.path.join(os.path.dirname(pg), 'track.json')
    info = json.load(open(tj, encoding='utf-8'))
    info.update(video_sha256='v' * 64, session_sha256='s1')
    json.dump(info, open(tj, 'w', encoding='utf-8'))
    rows = f2_rows(pg, tg)
    b = X.theo_cho(dict(goc=pg), fake_a6(), rows, 'T1')['cho']['goc']
    assert b['trang_thai'] == 'chua_ket_luan' and any('clip_sha256' in w for w in b['ly_do'])
    for r in rows:
        r['clip_sha256'] = 'v' * 64
    assert X.theo_cho(dict(goc=pg), fake_a6(), rows, 'T1')['cho']['goc']['trang_thai'] == 'dat'


def test_f4_refusals(tmp_path):
    pg, tg = place(tmp_path, 'goc', (0.004, -0.0025), (1.0, 0.0))
    with pytest.raises(X.XError, match='per E place'):
        X.theo_cho(dict(K2=pg), fake_a6(), f2_rows(pg, tg), 'T1')
    with pytest.raises(X.XError, match='A6 file'):
        X.theo_cho(dict(goc=pg), dict(fake_a6(), _sha256=None), f2_rows(pg, tg), 'T1')
    p = write_rows(tmp_path, [dict(schema='x')])
    with pytest.raises(X.XError, match='cua_so_v1'):
        X.read_cua_so([p])


def test_f4_readers_accept_it(tmp_path):
    # CAM c7_chuoi_khung.doc_xoay, NHACTAM c7_nhac_tam.doc_theo_cho, HIEUCHUAN c8_hc_vet d_vet_m, C8B ghep_q's check
    import c7_chuoi_khung as CK
    import c7_nhac_tam as NT
    pg, tg = place(tmp_path, 'goc', (0.004, -0.0025), (1.0, 0.0))
    pk, tk = place(tmp_path, 'K1', (0.004, -0.0025), (0.2, 1.1), seed=2)
    d = X.theo_cho(dict(goc=pg, K1=pk), fake_a6(), f2_rows(pg, tg) + f2_rows(pk, tk, 'K1'), 'T1')
    p = tmp_path / 'xoay_v1.json'
    p.write_text(json.dumps(d, allow_nan=False), encoding='utf-8')
    for cho in ('goc', 'K1'):
        r = CK.doc_xoay(str(p), cho)
        assert r['ok'] and r['q'] == d['cho'][cho]['q_tag_m'] and r['a6_sha256'] == 'a' * 64, r
    m, _ = NT.doc_theo_cho(str(p), 'xoay_v1')
    assert set(m) == {'goc', 'K1'} and all(m[c]['r_floor_m'] is not None and m[c]['trang_thai'] == 'dat' for c in m)
    g = X.ghep_q({}, 'goc', d)
    assert not any('not xoay_v1' in w for w in g['ly_do'])          # the F4 file passes the xoay part of ghep_q
    assert not any('xoay goc' in w for w in g['ly_do'])


def test_f4_cli(tmp_path):
    pg, tg = place(tmp_path, 'goc', (0.004, -0.0025), (1.0, 0.0))
    pk, tk = place(tmp_path, 'K1', (0.004, -0.0025), (0.2, 1.1), seed=2)
    cs = write_rows(tmp_path, f2_rows(pg, tg) + f2_rows(pk, tk, 'K1'))
    a6p = tmp_path / 'a6.json'
    a6 = fake_a6()
    a6.pop('_sha256')
    a6p.write_text(json.dumps(a6), encoding='utf-8')
    out = tmp_path / 'x.json'
    base = ['--cho', f'goc={pg}', '--cho', f'K1={pk}', '--a6', str(a6p), '--out', str(out)]
    assert X.main(base) == 2 and not out.exists()                      # F4 needs --cua-so and --tam-id
    assert X.main(base + ['--cua-so', cs, '--tam-id', 'T1']) == 0
    d = json.loads(out.read_text(encoding='utf-8'))
    assert d['version'] == 'xoay_v1' and {n['vai'] for n in d['nguon']} >= {'a6', 'cua_so_v1', 'frames goc'}
    assert d['cho']['goc']['trang_thai'] == 'dat' and d['script']['commit']
    assert X.main(base + ['--cua-so', cs, '--tam-id', 'T1']) == 2      # written once


def test_f4_u_q_parts_and_a6_term(tmp_path):
    pg, tg = place(tmp_path, 'goc', (0.004, -0.0025), (1.0, 0.0))
    a6 = dict(fake_a6(), eps_g=0.02)                                   # a large A6 residual: its term is visible
    b = X.theo_cho(dict(goc=pg), a6, f2_rows(pg, tg), 'T1')['cho']['goc']
    u = b['u_q_phan']
    assert u['a6du_rad'] == pytest.approx(A6.a6du_rad(a6, u['rho_max_m']))
    assert u['a6_m'] == pytest.approx(math.hypot(*b['q_tag_m']) * u['a6du_rad']) and u['a6_m'] > 0
    assert b['u_q_m'] == pytest.approx(u['spread_m'] + u['se3_m'] + u['res_m'] + u['a6_m'])
