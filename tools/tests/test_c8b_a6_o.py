"""c8b_a6_o (cells of the camera-IMU check, design C8B v7 2b, 7c): read back by c8b_a6.kiem_o."""
import csv
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import c8b_a6 as A  # noqa: E402
import c8b_a6_o as O  # noqa: E402

KHOA = 'k' * 64
CHO = ['a' * 64]
LOCK = dict(tap=dict(khop=['c1'], kiem=['c2'], dich=['r1']))


def _rows(n=12, d=0.002, up=0.003, ui=0.002, tap='kiem', luot='c2', nodes=('0', '1'), x=0.1, missing=(), s0=100):
    out = []
    for i in range(n):
        for nd in nodes:
            out.append(dict(luot=luot, tap=tap, id=str(s0 + i), nut=nd, x_m=x, y_m=0.2, psi_rad=0.3, u_mps=0.4,
                            lan_lap_id='L1', d_rad='' if (i, nd) in missing else (d if nd == '0' else -d / 2),
                            U_psi_rad=up, U_imu_rad=ui, controller='PID'))
    return out


def _parse(rows, tmp_path, name='rows.csv'):
    p = tmp_path / name
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(O.COLS) + ['controller'])
        w.writeheader()
        w.writerows(rows)
    return O.read_rows(str(p))


def _all(tmp_path, kiem, extra=()):
    return _parse(kiem + _rows(n=3, tap='khop', luot='c1') + _rows(n=3, tap='dich', luot='r1') + list(extra), tmp_path)


def test_cell_dat_read_by_kiem_o(tmp_path):
    rows = _all(tmp_path, _rows())
    o = O.build(rows, 'giu_ngoai_a6', 'cho', KHOA, CHO)
    assert o['trang_thai'] == 'dat' and len(o['o']) == 1 and o['version'] == 'c8b_a6_o_v1'   # GATE reads version
    c = o['o'][0]
    assert c['n_id'] == 12 and c['n_thieu'] == 0 and c['ty_le_max'] == pytest.approx(0.4) and c['dat']
    assert c['bao_cao_controller']['PID']['max_abs'] == pytest.approx(0.002)
    ok, why = A.kiem_o(o, CHO, KHOA, rows, LOCK)
    assert ok, why
    A._check_sha(o, 'o')


def test_kiem_o_rebuilds_cells_and_checks_the_lock(tmp_path):
    # A6C2-2: a slipping cell removed (sha recomputed), empty khop / dich, sets not the lock, no rows -> not dat
    rows = _all(tmp_path, _rows() + _rows(x=0.7, d=0.006, luot='c2', s0=300))
    o = O.build(rows, 'giu_ngoai_a6', 'cho', KHOA, CHO)
    assert o['trang_thai'] == 'truot' and len(o['o']) == 2
    cut = A.with_sha(dict(o, o=[c for c in o['o'] if c['dat']], trang_thai='dat'))
    ok, why = A.kiem_o(cut, CHO, KHOA, rows, LOCK)
    assert not ok and any('rebuilt' in w for w in why), why
    rows1 = _all(tmp_path, _rows())
    o1 = O.build(rows1, 'giu_ngoai_a6', 'cho', KHOA, CHO)
    for bad, txt in ((A.with_sha(dict(o1, tap=dict(o1['tap'], khop=[], dich=[]))), 'non-empty'),):
        ok, why = A.kiem_o(bad, CHO, KHOA, rows1, dict(tap=dict(khop=[], kiem=['c2'], dich=[])))
        assert not ok and any(txt in w for w in why), why
    for lock in (None, {}, dict(tap=dict(khop=['c1'], kiem=['c2', 'c9'], dich=['r1']))):
        ok, why = A.kiem_o(o1, CHO, KHOA, rows1, lock)
        assert not ok and any('lock' in w for w in why), why
    ok, why = A.kiem_o(o1, CHO, KHOA, None, LOCK)
    assert not ok and any('not rebuilt' in w for w in why)


def test_cell_over_bound_any_node_is_truot(tmp_path):
    rows = _rows()
    rows[5]['d_rad'] = 0.006                                            # one id at one node: |d| > U
    o = O.build(_all(tmp_path, rows), 'giu_ngoai_a6', 'cho', KHOA, CHO)
    assert o['trang_thai'] == 'truot' and o['o'][0]['ty_le_max'] == pytest.approx(1.2)
    assert o['o'][0]['d_max_rad'] == pytest.approx(0.006) and not o['o'][0]['dat']
    ok, why = A.kiem_o(o, CHO, KHOA, _all(tmp_path, rows), LOCK)
    assert not ok and any('U_psi' in w for w in why)


def test_id_counts_only_when_every_node_compares(tmp_path):
    rows = _rows(missing={(3, '1')})                                    # id 103 lost at node 1: thieu, not counted
    o = O.build(_all(tmp_path, rows), 'giu_ngoai_a6', 'cho', KHOA, CHO)
    c = o['o'][0]
    assert c['n_id'] == 11 and c['n_thieu'] == 1 and c['n_tho'] == 12 and not c['dat'] and o['trang_thai'] == 'thieu'
    assert not A.kiem_o(o, CHO, KHOA, _all(tmp_path, rows), LOCK)[0]
    r2 = _all(tmp_path, _rows(n=9))
    o2 = O.build(r2, 'giu_ngoai_a6', 'cho', KHOA, CHO)     # n_id < 10
    assert o2['trang_thai'] == 'thieu' and not A.kiem_o(o2, CHO, KHOA, r2, LOCK)[0]


def test_cells_sets_and_lock(tmp_path):
    rows = _rows() + _rows(x=0.7, luot='c3')                            # two zones -> two cells
    o = O.build(_all(tmp_path, rows), 'giu_ngoai_a6', 'cho', KHOA, CHO)
    assert len(o['o']) == 2 and o['tap']['kiem'] == ['c2', 'c3']
    assert O.cell_of(0.1, 0.2, 0.3, 0.4, 'L1') == (0, 0, 0, 1, 'L1') and O.cell_of(0, 0, 0, 2.0, 'L1') is None
    assert O.cell_of(0, 0, math.radians(46), 0.0, 'L1')[2] == 1
    with pytest.raises(O.OError, match='two sets'):
        O.build(_parse(_rows() + _rows(n=3, tap='khop', luot='c2', s0=500), tmp_path), 'giu_ngoai_a6', 'cho', KHOA, CHO)
    with pytest.raises(O.OError, match='khoa'):
        O.build(_all(tmp_path, _rows()), 'giu_ngoai_a6', 'cho', None, CHO)
    o3 = O.build(_all(tmp_path, _rows()), 'd2', 'cho', KHOA, CHO)        # D2: the cells of the target runs
    assert o3['o'][0]['tap'] == 'dich' and o3['o'][0]['n_id'] == 3


def test_cli(tmp_path):
    p = tmp_path / 'rows.csv'
    rows = _rows() + _rows(n=3, tap='khop', luot='c1') + _rows(n=3, tap='dich', luot='r1')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(O.COLS) + ['controller'])
        w.writeheader()
        w.writerows(rows)
    k = tmp_path / 'khoa.json'
    k.write_text(json.dumps(LOCK), encoding='utf-8')
    c = tmp_path / 'a6_goc.json'
    c.write_text('{}', encoding='utf-8')
    out = tmp_path / 'o.json'
    assert O.main(['--rows', str(p), '--muc-dich', 'giu_ngoai_a6', '--giu-ngoai-theo', 'cho', '--khoa-truoc', str(k),
                   '--cho-file', str(c), '--out', str(out)]) == 0
    d = json.loads(out.read_text(encoding='utf-8'))
    A._check_sha(d, 'o')
    import c8_u_imu as cu
    assert A.kiem_o(d, [cu.sha256(str(c))], cu.sha256(str(k)), O.read_rows(str(p)), LOCK)[0]
    rows[0]['U_psi_rad'] = -1
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(O.COLS) + ['controller'])
        w.writeheader()
        w.writerows(rows)
    assert O.main(['--rows', str(p), '--muc-dich', 'giu_ngoai_a6', '--giu-ngoai-theo', 'cho', '--khoa-truoc', str(k),
                   '--out', str(tmp_path / 'o2.json')]) == 2


def test_d2_mask_cells_not_whole_file(tmp_path):
    """D2 (design v7 7c line 166; CODE 24/11 khuya): a cell over the bound or short of samples masks its own frames
    only; the file stays dat with the mask (o_dat, o_thieu, ngoai_dai_u). The A6 held-out branch keeps its strict rule."""
    dich = (_rows(n=12, tap='dich', luot='r1')                                  # cell A: dat
            + _rows(n=12, tap='dich', luot='r1', x=0.7, d=0.006, s0=200)        # cell B: over (0.006 > 0.005)
            + _rows(n=4, tap='dich', luot='r1', x=1.3, s0=300)                  # cell C: 4 ids < N_ID_MIN
            + _rows(n=12, tap='dich', luot='r1', x=1.9, s0=400, missing={(0, '1')})  # cell D: one id not compared
            + [dict(r, u_mps=2.5) for r in _rows(n=2, tap='dich', luot='r1', s0=600)])  # outside the u bands
    rows = _parse(dich + _rows(n=3, tap='khop', luot='c1') + _rows(n=3, tap='kiem', luot='c2'), tmp_path)
    o = O.build(rows, 'd2', 'cho', KHOA, CHO)
    assert o['trang_thai'] == 'dat'
    m = o['mat_na']
    ids = {c['o_id']: c for c in o['o']}
    a = [k for k, c in ids.items() if c['vung'][0] == 0][0]
    assert m['o_dat'] == [a]
    why = {x['o_id']: x['ly_do'] for x in m['o_thieu']}
    assert sorted(map(tuple, why.values())) == [('thieu_id',), ('thieu_mau',), ('vuot',)]
    assert m['ngoai_dai_u'] == [['r1', 600], ['r1', 601]]
    # the A6 held-out check on the same kind of cells is still strict
    rk = _parse(_rows(n=3, tap='khop', luot='c1') + _rows(n=12, tap='kiem', luot='c2')
                + _rows(n=12, tap='kiem', luot='c2', x=0.7, d=0.006, s0=200) + _rows(n=3, tap='dich', luot='r1'),
                tmp_path, 'rk.csv')
    ok = O.build(rk, 'giu_ngoai_a6', 'cho', KHOA, CHO)
    assert ok['trang_thai'] == 'truot' and 'mat_na' not in ok
    assert O.build(_parse(_rows(n=3, tap='khop', luot='c1') + _rows(n=3, tap='kiem', luot='c2'), tmp_path, 'e.csv'),
                   'd2', 'cho', KHOA, CHO)['trang_thai'] == 'thieu'           # no target run, no cell
