"""Tests of c8b_d3_hang (D3 rows from the source files; CODE 24/11 'lam D3 truoc')."""
import csv
import hashlib
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import c8b_d3_hang as H        # noqa: E402
import c8b_kiem_vi_tri as KV   # noqa: E402


def _w(p, d):
    p.write_text(json.dumps(d), encoding='utf-8')
    return str(p), hashlib.sha256(p.read_bytes()).hexdigest()


def _setup(tmp_path, f_sau_trang_thai='chan_doan'):
    a6, a6s = _w(tmp_path / 'a6.json', dict(mien=dict(rho_max_m=1.6)))
    uh, uhs = _w(tmp_path / 'uh.json', dict(schema='c8b_u_h_chung_v1', version='c8b_u_h_v1', trang_thai='dat', u_h_chung_m=0.00825,
                                            giu_ngoai=[dict(ten='A', sai_m=0.0022), dict(ten='A2', sai_m=-0.0029)]))
    f0, f0s = _w(tmp_path / 'f7_0.json', dict(schema='truc_xe_v1', cho='goc', trang_thai='khoa', a_k_deg=0.30,
                                              u_a_noi_bo_deg=0.36, session_sha256='1' * 64))
    f1, _ = _w(tmp_path / 'f7_1.json', dict(schema='truc_xe_v1', cho='goc', trang_thai=f_sau_trang_thai,
                                            a_k_deg=0.33, u_a_noi_bo_deg=None, session_sha256='2' * 64))
    d2, _ = _w(tmp_path / 'dr2.json', dict(kind='c7_calib_drift', D_H_mm=0.29, eps_H_mm=2.0, thieu_H1=False))
    kh, _ = _w(tmp_path / 'khoa.json', dict(version='c8b_khoa_v5', trang_thai='khoa', le=1.25,
                                            eps_H=dict(mm=2.0, huong_deg=0.15),
                                            khoa_truoc=dict(nguon=[dict(vai='a6', sha256=a6s),
                                                                   dict(vai='u_h_chung', sha256=uhs),
                                                                   dict(vai='truc_xe', sha256=f0s)])))
    return dict(a6=a6, uh=uh, f0=f0, f1=f1, d2=d2, kh=kh)


def _argv(p, out, **kw):
    a = ['--khoa', p['kh'], '--a6', p['a6'], '--u-h', p['uh'], '--drift', f'2026-11-20_run02/doan3={p["d2"]}',
         '--goc-a-dau', p['f0'], '--goc-a', f'2026-11-23_run03={p["f1"]}', '--out', str(out)]
    return a


def test_rows_and_kiem_vi_tri(tmp_path, capsys):
    p = _setup(tmp_path)
    out = tmp_path / 'd3.csv'
    assert H.main(_argv(p, out)) == 0 and capsys.readouterr().out.rstrip().endswith('#ket_qua,ma=0')
    rows = list(csv.DictReader(open(out, encoding='utf-8')))
    by = {}
    for r in rows:
        by.setdefault(r['nguon'], []).append(r)
    assert [float(r['sai_m']) for r in by['do_nhay_cao']] == [0.0022, -0.0029]
    assert all(float(r['chi_so_m']) == 0.00825 and r['tap'] == 'kiem' for r in by['do_nhay_cao'])
    (t,) = by['troi_H']
    assert float(t['sai_m']) == pytest.approx(0.00029) and float(t['chi_so_m']) == pytest.approx(0.002)
    assert t['mien'] == '2026-11-20_run02/doan3'
    (g,) = by['dau_cuoi_lan_lap']
    assert float(g['sai_m']) == pytest.approx(1.6 * math.radians(0.03))
    assert float(g['chi_so_m']) == pytest.approx(1.6 * math.radians(0.36)) and g['co_chuan_dong'] == '0'
    meta = json.load(open(str(out) + '.nguon.json', encoding='utf-8'))
    assert meta['nguon_chua'] == ['P1_A3_H32', 'replay_neo', 'E']
    assert meta['thong_tin']['dau_cuoi_lan_lap']['gioi_han_dau_sau'] and 'HC_goc' in meta['liam_dau_cuoi_lan_lap']
    # the consumer reads it; the four missing sources keep the file thieu
    k = json.load(open(p['kh'], encoding='utf-8'))
    d3 = KV.build(KV.read_rows(str(out)), k, 'x' * 64)
    assert d3['trang_thai'] == 'thieu' and 'P1_A3_H32' in d3['ly_do'][0]
    assert {(b['nguon'], b['trang_thai']) for b in d3['bang']} == {('do_nhay_cao', 'gioi_han'), ('troi_H', 'gioi_han'),
                                                                   ('dau_cuoi_lan_lap', 'gioi_han')}
    assert H.main(_argv(p, out)) == 2                                       # never overwrites


@pytest.mark.parametrize('bad', ['uh_not_lock', 'a6_not_lock', 'f7_not_lock', 'f7_dau_not_khoa', 'same_session',
                                 'drift_no_H1', 'other_place', 'pair'])
def test_refused(tmp_path, bad):
    p = _setup(tmp_path)
    if bad == 'uh_not_lock':
        p['uh'], _ = _w(tmp_path / 'uh2.json', dict(schema='c8b_u_h_chung_v1', version='c8b_u_h_v1', trang_thai='dat', u_h_chung_m=0.01, giu_ngoai=[dict(sai_m=0.0)]))
    elif bad == 'a6_not_lock':
        p['a6'], _ = _w(tmp_path / 'a62.json', dict(mien=dict(rho_max_m=1.7)))
    elif bad == 'f7_not_lock':
        p['f0'], _ = _w(tmp_path / 'f7x.json', dict(schema='truc_xe_v1', cho='goc', trang_thai='khoa', a_k_deg=0.3,
                                                     u_a_noi_bo_deg=0.36, session_sha256='1' * 64, x=1))
    elif bad == 'f7_dau_not_khoa':
        d = json.load(open(p['f0'], encoding='utf-8'))
        d['trang_thai'] = 'chan_doan'
        f, s = _w(tmp_path / 'f7_0.json', d)
        k = json.load(open(p['kh'], encoding='utf-8'))
        k['khoa_truoc']['nguon'][2]['sha256'] = s
        _w(tmp_path / 'khoa.json', k)
    elif bad == 'same_session':
        p['f1'], _ = _w(tmp_path / 'f7_1.json', dict(schema='truc_xe_v1', cho='goc', trang_thai='khoa', a_k_deg=0.3,
                                                      session_sha256='1' * 64))
    elif bad == 'drift_no_H1':
        p['d2'], _ = _w(tmp_path / 'dr2.json', dict(kind='c7_calib_drift', D_H_mm=None, thieu_H1=True))
    elif bad == 'other_place':
        p['f1'], _ = _w(tmp_path / 'f7_1.json', dict(schema='truc_xe_v1', cho='K1', a_k_deg=0.3, session_sha256='2' * 64))
    out = tmp_path / 'd3.csv'
    a = _argv(p, out)
    if bad == 'pair':
        i = a.index('--goc-a')
        a = a[:i] + a[i + 2:]
    assert H.main(a) == 2 and not out.exists()


def _kc(tmp_path, seg, p1, a3, pairs, quys):
    """a3: A3_xp_deg.goc, the A3 number the lock takes (the index of the other sessions); A3loc_max_deg is the largest
    quy of a3_cap.csv, and the JSON records the SHA of cap_tuong_doi.csv, as KIEMCHEO writes them (D3-7)."""
    d = tmp_path / seg.replace('/', '_')
    d.mkdir()
    with open(d / 'cap_tuong_doi.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['clip', 'vung', 's', 'j', 'd_x', 'd_y'])
        w.writeheader()
        for i, (dx, dy) in enumerate(pairs):
            w.writerow(dict(clip='H_a', vung='goc', s=25, j=i, d_x=dx, d_y=dy))
            w.writerow(dict(clip='H_a', vung='goc', s=25, j=i, d_x=dx, d_y=dy))      # repeated (phi x nhin): once
    with open(d / 'a3_cap.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['clip', 'i', 'j', 'quy_deg'])
        w.writeheader()
        for i, q in enumerate(quys):
            w.writerow(dict(clip='H_a', i=1, j=i + 2, quy_deg=q))
    cs = hashlib.sha256((d / 'cap_tuong_doi.csv').read_bytes()).hexdigest()
    kc, s = _w(d / 'kiem_cheo.json', dict(P1_tuong_doi=dict(theo_vung=dict(goc=dict(max_m=p1)), cap_csv=dict(sha256=cs)),
                                          A3_tuong_doi=dict(A3_xp_deg=dict(goc=a3), A3loc_max_deg=max(quys))))
    return kc, s


def test_p1a3_held_out_by_session_and_e(tmp_path):
    p = _setup(tmp_path)
    k1, s1 = _kc(tmp_path, '2026-11-20_run02/doan2', 0.004, 0.2, [(0.003, 0.0), (0.0, -0.002)], [0.15, 0.2])
    k2, s2 = _kc(tmp_path, '2026-11-23_run03/doan1', 0.0045, 0.25, [(0.0044, 0.0)], [0.25])
    a6 = json.load(open(p['a6'], encoding='utf-8'))
    a6['kiem_imu'] = [dict(luot_id_E='HC_goc_1', cho='goc', du_rad=0.00066, nguong_rad=0.0031)]
    p['a6'], a6s = _w(tmp_path / 'a6.json', a6)
    k = json.load(open(p['kh'], encoding='utf-8'))
    k['khoa_truoc']['nguon'] = [n for n in k['khoa_truoc']['nguon'] if n['vai'] != 'a6'] + [
        dict(vai='a6', sha256=a6s), dict(vai='kiem_cheo', sha256=s1), dict(vai='kiem_cheo', sha256=s2)]
    _w(tmp_path / 'khoa.json', k)
    out = tmp_path / 'd3.csv'
    argv = _argv(p, out) + ['--kiem-cheo', f'2026-11-20_run02/doan2={k1}', '--kiem-cheo',
                            f'2026-11-23_run03/doan1={k2}', '--e']
    assert H.main(argv) == 0
    rows = list(csv.DictReader(open(out, encoding='utf-8')))
    p1 = [r for r in rows if r['nguon'] == 'P1_A3_H32' and r['dai_luong'] == 'P1']
    assert len(p1) == 3                                                    # repeated pair rows taken once
    r2 = [r for r in p1 if r['mien'] == '2026-11-20_run02/doan2:goc']
    assert all(float(r['chi_so_m']) == pytest.approx(1.25 * 0.0045) for r in r2)   # the OTHER session x le (L155)
    r3 = [r for r in p1 if r['mien'] == '2026-11-23_run03/doan1:goc']
    assert float(r3[0]['chi_so_m']) == pytest.approx(1.25 * 0.004) and float(r3[0]['sai_m']) == pytest.approx(0.0044)
    a3 = [r for r in rows if r['dai_luong'] == 'A3']
    assert {r['mien']: float(r['chi_so_m']) for r in a3} == pytest.approx(
        {'2026-11-20_run02/doan2': 1.25 * 1.6 * math.radians(0.25),
         '2026-11-23_run03/doan1': 1.25 * 1.6 * math.radians(0.2)})
    (e,) = [r for r in rows if r['nguon'] == 'E']
    assert float(e['sai_m']) == pytest.approx(1.6 * 0.00066) and e['mien'] == 'goc'
    d3 = KV.build(KV.read_rows(str(out)), k, 'x' * 64)
    assert 'replay_neo' in d3['ly_do'][0] and 'P1_A3_H32' not in d3['ly_do'][0]
    # L155: the run03 pair 4.4 mm is inside the session-2 index 4.0 mm x the margin 1.25 (thieu under the first rule)
    assert not any(m['dai_luong'] == 'P1' for m in d3['mien_thieu'])
    # one session only, or a kiem_cheo the lock does not name: refused
    assert H.main(_argv(p, tmp_path / 'b.csv') + ['--kiem-cheo', f'2026-11-20_run02/doan2={k1}']) == 2
    k3, _ = _kc(tmp_path, '2026-11-24_run05/doan1', 0.0041, 0.21, [(0.001, 0.0)], [0.21])   # not named by the lock
    assert H.main(_argv(p, tmp_path / 'c.csv') + ['--kiem-cheo', f'2026-11-20_run02/doan2={k1}', '--kiem-cheo',
                                                  f'2026-11-24_run05/doan1={k3}']) == 2


def _neo(tmp_path, b, P, A, frames):
    d = tmp_path / f'neo_{b}'
    d.mkdir()
    nj, s = _w(d / 'neo_luot.json', dict(luot={f'L_{b}': dict(cho='goc', P_neo_m=P, A_neo_rad=A)},
                                         theo_cho=dict(goc=dict(P_neo_m=P, A_neo_rad=A))))
    kc = d / 'khung.csv'
    with open(kc, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['luot_id', 'khung', 't_s', 'la_W', 'sau_W', 'trang_thai', 'dx_m', 'dy_m',
                                          'dpsi_rad', 'n_R', 'ly_do'])
        w.writeheader()
        for i, (st, dx, dy, dp) in enumerate(frames):
            w.writerow(dict(luot_id=f'L_{b}', khung=i, t_s=i / 60, la_W=0, sau_W=1, trang_thai=st, dx_m=dx, dy_m=dy,
                            dpsi_rad=dp, n_R=8, ly_do='' if st == 'hop_le' else 'mat_W'))
    return nj, s, str(kc)


def test_replay_neo_held_out_by_session(tmp_path):
    p = _setup(tmp_path)
    n1, s1, c1 = _neo(tmp_path, '2026-11-20_run02', 3e-5, 1e-5, [('hop_le', 3e-5, 0.0, 1e-5), ('mat_W', '', '', '')])
    n2, s2, c2 = _neo(tmp_path, '2026-11-23_run03', 4.5e-5, 2e-5, [('hop_le', 0.0, 4.5e-5, -2e-5)])
    k = json.load(open(p['kh'], encoding='utf-8'))
    k['khoa_truoc']['nguon'] += [dict(vai='neo', sha256=s1), dict(vai='neo', sha256=s2)]
    _w(tmp_path / 'khoa.json', k)
    out = tmp_path / 'd3.csv'
    argv = _argv(p, out) + ['--neo', f'2026-11-20_run02={n1},{c1}', '--neo', f'2026-11-23_run03={n2},{c2}']
    assert H.main(argv) == 0
    rows = [r for r in csv.DictReader(open(out, encoding='utf-8')) if r['nguon'] == 'replay_neo']
    pr = {r['mien']: (float(r['sai_m']), float(r['chi_so_m'])) for r in rows if r['dai_luong'] == 'P_neo'}
    assert pr['2026-11-20_run02:goc'] == pytest.approx((3e-5, 1.25 * 4.5e-5))    # the OTHER session x le (L155)
    assert pr['2026-11-23_run03:goc'] == pytest.approx((4.5e-5, 1.25 * 3e-5))
    ar = [r for r in rows if r['dai_luong'] == 'A_neo' and r['mien'] == '2026-11-23_run03:goc']
    assert float(ar[0]['sai_m']) == pytest.approx(1.6 * 2e-5) and float(ar[0]['chi_so_m']) == pytest.approx(1.25 * 1.6 * 1e-5)
    meta = json.load(open(str(out) + '.nguon.json', encoding='utf-8'))
    assert meta['thong_tin']['khung_thieu_replay_neo'] == [['2026-11-20_run02', 'L_2026-11-20_run02', '1', 'mat_W',
                                                             'mat_W']]
    assert H.main(_argv(p, tmp_path / 'b.csv') + ['--neo', f'2026-11-20_run02={n1},{c1}']) == 2   # one session
    # the CSV of another neo_luot.json (its frames do not give P_neo_m, A_neo_rad of the lock's file): refused
    assert H.main(_argv(p, tmp_path / 'c.csv') + ['--neo', f'2026-11-20_run02={n1},{c2}',
                                                  '--neo', f'2026-11-23_run03={n2},{c2}']) == 2


def test_he_so_giu_ngoai_from_the_ratios(tmp_path):
    """c8b_he_so_giu_ngoai: ratio = largest |error| of a session / index of the others; above the margin 1.25 the
    factor is ratio / 1.25, else 1; an A3 ratio above the margin is thieu (A3 is not scaled by the lock)."""
    import c8b_he_so_giu_ngoai as HS
    k1, _ = _kc(tmp_path, '2026-11-20_run02/doan2', 0.003, 0.2, [(0.003, 0.0), (0.0, -0.002)], [0.2])
    k2, _ = _kc(tmp_path, '2026-11-23_run03/doan1', 0.0045, 0.21, [(0.0045, 0.0)], [0.21])
    n1, _, c1 = _neo(tmp_path, '2026-11-20_run02', 3e-5, 1e-5, [('hop_le', 3e-5, 0.0, 1e-5)])
    n2, _, c2 = _neo(tmp_path, '2026-11-23_run03', 3.3e-5, 2e-5, [('hop_le', 0.0, 3.3e-5, -2e-5)])
    out = tmp_path / 'hs.json'
    argv = ['--kiem-cheo', f'2026-11-20_run02/doan2={k1}', '--kiem-cheo', f'2026-11-23_run03/doan1={k2}',
            '--neo', f'2026-11-20_run02={n1},{c1}', '--neo', f'2026-11-23_run03={n2},{c2}', '--out', str(out)]
    assert HS.main(argv) == 0
    d = json.load(open(out, encoding='utf-8'))
    assert d['trang_thai'] == 'dat' and d['le'] == 1.25
    assert d['ti_le']['P1_m']['ti_le'] == pytest.approx(1.5) and d['hang']['P1_m'] == pytest.approx(1.2)
    assert d['ti_le']['P_neo_m']['ti_le'] == pytest.approx(1.1) and d['hang']['P_neo_m'] == 1.0
    assert d['ti_le']['A_neo_rad']['ti_le'] == pytest.approx(2.0) and d['hang']['A_neo_rad'] == pytest.approx(1.6)
    assert d['ti_le']['A3_rad']['ti_le'] == pytest.approx(1.05)
    import c8b_khoa_v5 as K
    ng = [dict(vai='neo', file='n', sha256=H.cu.sha256(n1)), dict(vai='kiem_cheo', file='k', sha256=H.cu.sha256(k2))]
    assert K.doc_he_so(d, ng) == d['hang']                                         # the lock reads it
    k3, _ = _kc(tmp_path, '2026-11-24_run05/doan1', 0.004, 0.3, [(0.004, 0.0)], [0.3])
    assert HS.main(argv[:-2] + ['--kiem-cheo', f'2026-11-24_run05/doan1={k3}', '--out', str(tmp_path / 'h2.json')]) == 0
    assert json.load(open(tmp_path / 'h2.json', encoding='utf-8'))['trang_thai'] == 'thieu'    # A3 0.3 / 0.21 > 1.25


def test_l155_factor_file_of_the_lock(tmp_path):
    """L155: a lock that names a he_so_giu_ngoai source needs --he-so (that file); the index of a scaled term is the
    other sessions' index x le x factor, so the largest error of the trial session is inside it."""
    import c8b_he_so_giu_ngoai as HS
    p = _setup(tmp_path)
    n1, s1, c1 = _neo(tmp_path, '2026-11-20_run02', 3e-5, 1e-5, [('hop_le', 3e-5, 0.0, 1e-5)])
    n2, s2, c2 = _neo(tmp_path, '2026-11-23_run03', 3.3e-5, 2e-5, [('hop_le', 0.0, 3.3e-5, -2e-5)])
    k1, ks1 = _kc(tmp_path, '2026-11-20_run02/doan2', 0.004, 0.2, [(0.004, 0.0)], [0.2])
    k2, ks2 = _kc(tmp_path, '2026-11-23_run03/doan1', 0.0045, 0.21, [(0.0045, 0.0)], [0.21])
    hs = tmp_path / 'hs.json'
    kcv = ['--kiem-cheo', f'2026-11-20_run02/doan2={k1}', '--kiem-cheo', f'2026-11-23_run03/doan1={k2}']
    neov = ['--neo', f'2026-11-20_run02={n1},{c1}', '--neo', f'2026-11-23_run03={n2},{c2}']
    assert HS.main(kcv + neov + ['--out', str(hs)]) == 0
    hd = json.load(open(hs, encoding='utf-8'))
    assert hd['hang']['A_neo_rad'] == pytest.approx(2.0 / 1.25) and hd['hang']['P1_m'] == 1.0
    k = json.load(open(p['kh'], encoding='utf-8'))
    k['khoa_truoc']['nguon'] += [dict(vai='neo', sha256=s1), dict(vai='neo', sha256=s2),
                                 dict(vai='kiem_cheo', sha256=ks1), dict(vai='kiem_cheo', sha256=ks2),
                                 dict(vai='he_so_giu_ngoai', sha256=H.cu.sha256(str(hs)))]
    _w(tmp_path / 'khoa.json', k)
    out = tmp_path / 'd3.csv'
    assert H.main(_argv(p, tmp_path / 'x.csv') + kcv + neov) == 2                  # the lock names a factor file
    assert H.main(_argv(p, out) + kcv + neov + ['--he-so', str(hs)]) == 0
    rows = [r for r in csv.DictReader(open(out, encoding='utf-8')) if r['dai_luong'] == 'A_neo']
    worst = max(rows, key=lambda r: abs(float(r['sai_m'])) / float(r['chi_so_m']))
    assert abs(float(worst['sai_m'])) <= float(worst['chi_so_m'])                   # inside, by the factor
    assert abs(float(worst['sai_m'])) / float(worst['chi_so_m']) == pytest.approx(1.0, abs=1e-8)
    import c8b_kiem_vi_tri as KV
    d3 = KV.build(KV.read_rows(str(out)), k, 'x' * 64)
    assert not [m for m in d3['mien_thieu'] if m['nguon'] in ('replay_neo', 'P1_A3_H32')]
    # Codex D3-3: c8b_kiem_vi_tri takes the rows c8b_d3_hang wrote for THIS lock only (the .nguon.json beside them)
    assert KV.main(['--rows', str(out), '--khoa', p['kh'], '--out', str(tmp_path / 'kvt.json')]) == 0
    kb, _ = _w(tmp_path / 'khoa_b.json', dict(k, ghi_chu='another lock'))
    assert KV.main(['--rows', str(out), '--khoa', kb, '--out', str(tmp_path / 'kvt_b.json')]) == 2
    with open(out, 'a', encoding='utf-8') as f:                                    # the rows edited after c8b_d3_hang
        f.write('E,kiem,kiem_imu_E,K9,0.0,0.001,0\n')
    assert KV.main(['--rows', str(out), '--khoa', p['kh'], '--out', str(tmp_path / 'kvt_c.json')]) == 2
    # Codex D3-4: the factor file is of exactly the files compared (labels and SHA); one file under two labels refused
    kcx = ['--kiem-cheo', f'2026-11-20_run02/doan9={k1}', '--kiem-cheo', f'2026-11-23_run03/doan1={k2}']
    assert H.main(_argv(p, tmp_path / 'y.csv') + kcx + neov + ['--he-so', str(hs)]) == 2
    kc2 = ['--kiem-cheo', f'2026-11-20_run02/doan2={k1}', '--kiem-cheo', f'2026-11-23_run03/doan1={k1}']
    assert HS.main(kc2 + neov + ['--out', str(tmp_path / 'hs2.json')]) == 2
    ne2 = ['--neo', f'2026-11-20_run02={n1},{c1}', '--neo', f'2026-11-23_run03={n1},{c1}']
    assert HS.main(kcv + ne2 + ['--out', str(tmp_path / 'hs3.json')]) == 2
    # Codex round 2 D3v2-4: two files under ONE label (the last would win, the order would decide the factor)
    kc3 = kcv + ['--kiem-cheo', f'2026-11-20_run02/doan2={k2}']
    assert HS.main(kc3 + neov + ['--out', str(tmp_path / 'hs4.json')]) == 2
    assert H.main(_argv(p, tmp_path / 'z.csv') + kc3 + neov + ['--he-so', str(hs)]) == 2


def test_pair_csv_tied_to_the_kiem_cheo_json(tmp_path):
    """Codex D3-7: cap_tuong_doi.csv by the SHA the JSON records, a3_cap.csv by its largest quy = A3loc_max_deg."""
    k1, _ = _kc(tmp_path, '2026-11-20_run02/doan2', 0.004, 0.2, [(0.004, 0.0)], [0.2, 0.1])
    kc, s, cap, cs, a3, as_ = H.doc_kiem_cheo(k1)
    assert len(cap) == 2 and len(a3) == 2
    d = os.path.dirname(k1)
    rows = open(os.path.join(d, 'a3_cap.csv'), encoding='utf-8').read().splitlines()
    with open(os.path.join(d, 'a3_cap.csv'), 'w', encoding='utf-8', newline='') as f:   # the largest pair taken out
        f.write('\n'.join([rows[0], rows[2]]) + '\n')
    with pytest.raises(H.HangError, match='a3_cap'):
        H.doc_kiem_cheo(k1)
    k3, _ = _kc(tmp_path, '2026-11-24_run05/doan1', 0.004, 0.30, [(0.004, 0.0)], [0.20])   # A3_xp above the pairs
    with pytest.raises(H.HangError, match='contradicts'):
        H.doc_kiem_cheo(k3)
    k2, _ = _kc(tmp_path, '2026-11-23_run03/doan1', 0.004, 0.2, [(0.004, 0.0)], [0.2])
    with open(os.path.join(os.path.dirname(k2), 'cap_tuong_doi.csv'), 'a', encoding='utf-8') as f:
        f.write('H_a,goc,25,9,0.001,0.0\n')
    with pytest.raises(H.HangError, match='cap_tuong_doi'):
        H.doc_kiem_cheo(k2)


def test_a3_index_is_the_number_the_lock_takes(tmp_path):
    """Codex D3-6: the A3 index of the other sessions is A3_xp_deg.<CHO_MUC> (what c8b_khoa_v5 puts in the lock), not
    the largest quy over every pair of the floor."""
    k1, _ = _kc(tmp_path, '2026-11-20_run02/doan2', 0.004, 0.20, [(0.004, 0.0)], [0.40])   # 0.40 outside goc
    k2, _ = _kc(tmp_path, '2026-11-23_run03/doan1', 0.004, 0.30, [(0.004, 0.0)], [0.30])
    segs = []
    for seg, k in (('2026-11-20_run02/doan2', k1), ('2026-11-23_run03/doan1', k2)):
        kc, _, cap, _, a3, _ = H.doc_kiem_cheo(k)
        segs.append((seg, kc, cap, a3))
    a = {r['mien']: (float(r['sai_m']), float(r['chi_so_m'])) for r in H.rows_p1a3(segs, 1.0) if r['dai_luong'] == 'A3'}
    assert a['2026-11-20_run02/doan2'] == pytest.approx((math.radians(0.40), math.radians(0.30)))
    assert a['2026-11-23_run03/doan1'] == pytest.approx((math.radians(0.30), math.radians(0.20)))


def test_factor_covers_its_own_rows_after_rounding(tmp_path):
    """Codex D3-5: P_neo 1.5e-7 and 2.0625e-7 m (ratio 1.375, factor 1.1): index x le x factor fell one rounding step
    below the largest error; the factor is raised until every row of c8b_d3_hang passes, at any rho."""
    import c8b_he_so_giu_ngoai as HS
    k1, _ = _kc(tmp_path, '2026-11-20_run02/doan2', 0.004, 0.2, [(0.004, 0.0)], [0.2])
    k2, _ = _kc(tmp_path, '2026-11-23_run03/doan1', 0.004, 0.2, [(0.004, 0.0)], [0.2])
    n1, _, c1 = _neo(tmp_path, '2026-11-20_run02', 1.5e-7, 1.5e-7, [('hop_le', 1.5e-7, 0.0, 1.5e-7)])
    n2, _, c2 = _neo(tmp_path, '2026-11-23_run03', 2.0625e-7, 2.0625e-7, [('hop_le', 0.0, 2.0625e-7, -2.0625e-7)])
    segs, neo = [], []
    for seg, k in (('2026-11-20_run02/doan2', k1), ('2026-11-23_run03/doan1', k2)):
        kc, _, cap, _, a3, _ = H.doc_kiem_cheo(k)
        segs.append((seg, kc, cap, a3))
    for b, nj, c in (('2026-11-20_run02', n1, c1), ('2026-11-23_run03', n2, c2)):
        neo.append((b, H._read(nj)[0], H._csv(c)[0]))
    d = HS.build(segs, neo)
    assert d['trang_thai'] == 'dat' and 1.1 <= d['hang']['P_neo_m'] <= 1.1 + 1e-8
    for rho in (1.0, 1.615, 0.37):
        rows, _ = H.rows_neo(neo, rho, 1.25, d['hang'])
        assert all(abs(float(r['sai_m'])) <= float(r['chi_so_m']) for r in rows), rho


# ── L165 (Liam 25/11 at CODE: "Bo kiem, ghi gioi han (Recommended)"): a place measured in one session only ──────

def _neo_cho(tmp_path, b, cho_frames):
    """neo_luot.json + frame CSV of session b with one run per place: cho_frames {cho: [(dx, dy, dpsi), ...]} (all
    valid); P_neo_m, A_neo_rad of a run / place = the largest of its frames (as KIEMCHEO writes them)."""
    d = tmp_path / f'neo_{b}'
    d.mkdir()
    luot, theo, rows = {}, {}, []
    for cho, fr in cho_frames.items():
        P, A = max(math.hypot(x, y) for x, y, _ in fr), max(abs(p) for _, _, p in fr)
        luot[f'L_{b}_{cho}'] = dict(cho=cho, P_neo_m=P, A_neo_rad=A)
        theo[cho] = dict(P_neo_m=P, A_neo_rad=A)
        rows += [dict(luot_id=f'L_{b}_{cho}', khung=i, t_s=i / 60, la_W=0, sau_W=1, trang_thai='hop_le', dx_m=x, dy_m=y,
                      dpsi_rad=p, n_R=8, ly_do='') for i, (x, y, p) in enumerate(fr)]
    nj, s = _w(d / 'neo_luot.json', dict(luot=luot, theo_cho=theo))
    kc = d / 'khung.csv'
    with open(kc, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return nj, s, str(kc)


def test_l165_place_of_one_session_gives_no_row_and_is_a_limit(tmp_path):
    import c8b_he_so_giu_ngoai as HS
    import c8b_khoa_v5 as K
    p = _setup(tmp_path)
    B2, B3 = '2026-11-20_run02', '2026-11-23_run03'
    # session 2: goc and K1, K2 (the K frames far larger than any goc frame); session run03: goc only
    n1, s1, c1 = _neo_cho(tmp_path, B2, dict(goc=[(3e-5, 0.0, 1e-5)], K1=[(9e-4, 0.0, 9e-4)], K2=[(0.0, 8e-4, -8e-4)]))
    n2, s2, c2 = _neo_cho(tmp_path, B3, dict(goc=[(0.0, 3.3e-5, -2e-5)]))
    k1, ks1 = _kc(tmp_path, f'{B2}/doan2', 0.004, 0.2, [(0.004, 0.0)], [0.2])
    k2, ks2 = _kc(tmp_path, f'{B3}/doan1', 0.0045, 0.21, [(0.0045, 0.0)], [0.21])
    kcv = ['--kiem-cheo', f'{B2}/doan2={k1}', '--kiem-cheo', f'{B3}/doan1={k2}']
    neov = ['--neo', f'{B2}={n1},{c1}', '--neo', f'{B3}={n2},{c2}']
    # the rows: the K frames give no row; the goc frames are held out as before
    mb = []
    items = [(B2, json.load(open(n1, encoding='utf-8')), list(csv.DictReader(open(c1, encoding='utf-8')))),
             (B3, json.load(open(n2, encoding='utf-8')), list(csv.DictReader(open(c2, encoding='utf-8'))))]
    rows, thieu = H.rows_neo(items, 1.0, 1.25, None, mb)
    assert mb == [f'{B2}:K1', f'{B2}:K2'] and thieu == []
    assert sorted({r['mien'] for r in rows}) == [f'{B2}:goc', f'{B3}:goc'] and len(rows) == 4
    assert H.rows_neo(items, 1.0, 1.25)[0] == rows                      # without the list: the same rows, no error
    # the factor file: computed with the neo of session 2 (the lock asks for it), the K frames outside the ratios
    hs = tmp_path / 'hs.json'
    assert HS.main(kcv + neov + ['--out', str(hs)]) == 0
    hd = json.load(open(hs, encoding='utf-8'))
    assert hd['ti_le']['P_neo_m']['ti_le'] == pytest.approx(1.1) and hd['ti_le']['P_neo_m']['mien'] == f'{B3}:goc'
    assert hd['ti_le']['A_neo_rad']['ti_le'] == pytest.approx(2.0) and hd['hang']['P_neo_m'] == 1.0
    assert hd['neo_cho_mot_buoi']['mien'] == [f'{B2}:K1', f'{B2}:K2'] and 'L165' in hd['neo_cho_mot_buoi']['luat']
    ng = [dict(vai='neo', file='n', sha256=s1), dict(vai='kiem_cheo', file='k', sha256=ks1)]
    assert K.doc_he_so(hd, ng) == hd['hang']                            # the lock of session 2 takes the file
    # D3: the rows file and its limit; c8b_kiem_vi_tri writes the limit, the state does not change
    k = json.load(open(p['kh'], encoding='utf-8'))
    k['khoa_truoc']['nguon'] += [dict(vai='neo', sha256=s1), dict(vai='neo', sha256=s2),
                                 dict(vai='kiem_cheo', sha256=ks1), dict(vai='kiem_cheo', sha256=ks2),
                                 dict(vai='he_so_giu_ngoai', sha256=H.cu.sha256(str(hs)))]
    _w(tmp_path / 'khoa.json', k)
    out = tmp_path / 'd3.csv'
    assert H.main(_argv(p, out) + kcv + neov + ['--he-so', str(hs)]) == 0
    rr = [r for r in csv.DictReader(open(out, encoding='utf-8')) if r['nguon'] == 'replay_neo']
    assert sorted({r['mien'] for r in rr}) == [f'{B2}:goc', f'{B3}:goc']
    meta = json.load(open(str(out) + '.nguon.json', encoding='utf-8'))
    assert meta['thong_tin']['replay_neo_cho_mot_buoi']['mien'] == [f'{B2}:K1', f'{B2}:K2']
    assert 'L165' in meta['thong_tin']['replay_neo_cho_mot_buoi']['luat'] and meta['thong_tin']['khung_thieu_replay_neo'] == []
    kv = tmp_path / 'kvt.json'
    assert KV.main(['--rows', str(out), '--khoa', p['kh'], '--out', str(kv)]) == 0
    d3 = json.load(open(kv, encoding='utf-8'))
    gh = [g for g in d3['gioi_han'] if 'L165' in g['ly_do']]
    assert sorted((g['mien'], g['dai_luong']) for g in gh) == [(f'{B2}:K1', 'A_neo'), (f'{B2}:K1', 'P_neo'),
                                                              (f'{B2}:K2', 'A_neo'), (f'{B2}:K2', 'P_neo')]
    assert not [m for m in d3['mien_thieu'] if m['nguon'] == 'replay_neo']
    d3b = KV.build(KV.read_rows(str(out)), k, 'x' * 64)
    assert d3['trang_thai'] == d3b['trang_thai'] and d3['ly_do'] == d3b['ly_do']    # a limit, not a reason


def test_l165_no_place_shared_gives_no_ratio(tmp_path):
    # every place of one session only: no replay_neo row at all, so no ratio and no factor file (as before L165 for a
    # file without rows); without a place of one session the file says so (empty list)
    import c8b_he_so_giu_ngoai as HS
    B2, B3 = '2026-11-20_run02', '2026-11-23_run03'
    k1, _ = _kc(tmp_path, f'{B2}/doan2', 0.004, 0.2, [(0.004, 0.0)], [0.2])
    k2, _ = _kc(tmp_path, f'{B3}/doan1', 0.0045, 0.21, [(0.0045, 0.0)], [0.21])
    kcv = ['--kiem-cheo', f'{B2}/doan2={k1}', '--kiem-cheo', f'{B3}/doan1={k2}']
    n1, _, c1 = _neo_cho(tmp_path, B2, dict(K1=[(3e-5, 0.0, 1e-5)]))
    n2, _, c2 = _neo_cho(tmp_path, B3, dict(goc=[(0.0, 3.3e-5, -2e-5)]))
    assert HS.main(kcv + ['--neo', f'{B2}={n1},{c1}', '--neo', f'{B3}={n2},{c2}', '--out', str(tmp_path / 'a.json')]) == 2
    sub = tmp_path / 's'
    sub.mkdir()
    n3, _, c3 = _neo_cho(sub, B2, dict(goc=[(3e-5, 0.0, 1e-5)]))
    assert HS.main(kcv + ['--neo', f'{B2}={n3},{c3}', '--neo', f'{B3}={n2},{c2}', '--out', str(tmp_path / 'b.json')]) == 0
    assert json.load(open(tmp_path / 'b.json', encoding='utf-8'))['neo_cho_mot_buoi'] == dict(mien=[], luat=None)
