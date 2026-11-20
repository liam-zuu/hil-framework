"""c8b_kiem_vi_tri (D3, design C8B v7 0.5, 8): the fields GATE reads and the state rule."""
import csv
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import c8b_kiem_vi_tri as V  # noqa: E402

KHOA = dict(version='c8b_khoa_v5', trang_thai='khoa')


def _rows(extra=()):
    r = []
    for ng in V.NGUON_CAN:
        r.append(dict(nguon=ng, tap='khop', dai_luong='p', mien='goc', sai_m='0.001', chi_so_m='0.002',
                      co_chuan_dong='1'))
        r.append(dict(nguon=ng, tap='kiem', dai_luong='p', mien='goc', sai_m='-0.0015', chi_so_m='0.002',
                      co_chuan_dong='1'))
    return r + list(extra)


def _parse(rows, tmp_path):
    p = tmp_path / 'rows.csv'
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(V.COLS))
        w.writeheader()
        w.writerows(rows)
    return V.read_rows(str(p)), p


def test_dat_and_gate_fields(tmp_path):
    rows, _ = _parse(_rows(), tmp_path)
    o = V.build(rows, KHOA, 's' * 64)
    assert o['version'] == 'c8b_kiem_vi_tri_v1' and o['trang_thai'] == 'dat' and o['khoa_truoc_sha256'] == 's' * 64
    assert all(b['trang_thai'] == 'dat' for b in o['bang']) and not o['mien_thieu']


def test_rule_test_set_excess_fit_raise_and_no_dynamic_reference(tmp_path):
    ex = [dict(nguon='E', tap='kiem', dai_luong='p', mien='K1', sai_m='0.003', chi_so_m='0.002', co_chuan_dong='1'),
          dict(nguon='12b', tap='khop', dai_luong='p', mien='K1', sai_m='-0.0025', chi_so_m='0.002', co_chuan_dong='1'),
          dict(nguon='12b', tap='kiem', dai_luong='p', mien='K1', sai_m='0.0024', chi_so_m='0.002', co_chuan_dong='1'),
          dict(nguon='do_nhay_cao', tap='kiem', dai_luong='h', mien='goc', sai_m='0.0001', chi_so_m='0.001',
               co_chuan_dong='0')]
    rows, _ = _parse(_rows(ex), tmp_path)
    o = V.build(rows, KHOA, 's' * 64)
    st = {(b['nguon'], b['dai_luong'], b['mien']): b for b in o['bang']}
    assert st[('E', 'p', 'K1')]['trang_thai'] == 'thieu'                  # after the lock: thieu, not raised
    assert o['mien_thieu'] == [dict(nguon='E', dai_luong='p', mien='K1', sai_max_m=0.003, chi_so_m=0.002)]
    assert st[('12b', 'p', 'K1')]['trang_thai'] == 'dat' and st[('12b', 'p', 'K1')]['chi_so_dung_m'] == 0.0025
    assert o['chi_so_nang'][0]['chi_so_nang_m'] == 0.0025                 # before the lock: raised to the largest
    assert st[('do_nhay_cao', 'h', 'goc')]['trang_thai'] == 'gioi_han' and o['gioi_han']
    assert o['trang_thai'] == 'dat'                                       # domains thieu go to the scoring


def test_thieu_without_lock_or_source(tmp_path):
    rows, _ = _parse(_rows(), tmp_path)
    assert V.build(rows, dict(KHOA, trang_thai='chua_khoa'), 's' * 64)['trang_thai'] == 'thieu'
    rows2, _ = _parse([r for r in _rows() if r['nguon'] != 'troi_H'], tmp_path)
    o = V.build(rows2, KHOA, 's' * 64)
    assert o['trang_thai'] == 'thieu' and any('troi_H' in w for w in o['ly_do'])
    with pytest.raises(V.ViTriError, match='more than one index'):
        rows3, _ = _parse(_rows([dict(nguon='E', tap='kiem', dai_luong='p', mien='goc', sai_m='0', chi_so_m='0.003',
                                      co_chuan_dong='1')]), tmp_path)
        V.build(rows3, KHOA, 's' * 64)


def test_cli(tmp_path):
    _, p = _parse(_rows(), tmp_path)
    k = tmp_path / 'khoa.json'
    k.write_text(json.dumps(KHOA), encoding='utf-8')
    out = tmp_path / 'd3.json'
    assert V.main(['--rows', str(p), '--khoa', str(k), '--out', str(out)]) == 0
    d = json.loads(out.read_text(encoding='utf-8'))
    import c8_u_imu as cu
    assert d['khoa_truoc_sha256'] == cu.sha256(str(k)) and d['trang_thai'] == 'dat'
    assert V.main(['--rows', str(p), '--khoa', str(k), '--out', str(out)]) == 2          # never overwritten
