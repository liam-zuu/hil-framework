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
    uh, uhs = _w(tmp_path / 'uh.json', dict(schema='c8b_u_h_v1', u_h_chung_m=0.00825,
                                            giu_ngoai=[dict(ten='A', sai_m=0.0022), dict(ten='A2', sai_m=-0.0029)]))
    f0, f0s = _w(tmp_path / 'f7_0.json', dict(schema='truc_xe_v1', cho='goc', trang_thai='khoa', a_k_deg=0.30,
                                              u_a_noi_bo_deg=0.36, session_sha256='1' * 64))
    f1, _ = _w(tmp_path / 'f7_1.json', dict(schema='truc_xe_v1', cho='goc', trang_thai=f_sau_trang_thai,
                                            a_k_deg=0.33, u_a_noi_bo_deg=None, session_sha256='2' * 64))
    d2, _ = _w(tmp_path / 'dr2.json', dict(kind='c7_calib_drift', D_H_mm=0.29, eps_H_mm=2.0, thieu_H1=False))
    kh, _ = _w(tmp_path / 'khoa.json', dict(version='c8b_khoa_v5', trang_thai='khoa', eps_H=dict(mm=2.0, huong_deg=0.15),
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
    assert meta['nguon_chua'] == ['P1_A3_H32', 'replay_neo', '12b', 'E']
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
        p['uh'], _ = _w(tmp_path / 'uh2.json', dict(schema='c8b_u_h_v1', u_h_chung_m=0.01, giu_ngoai=[dict(sai_m=0.0)]))
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
