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
        w = csv.DictWriter(f, fieldnames=list(O.COLS) + ['controller', 'cham_id'], restval='')
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
    rows_d2 = [dict(r, cham_id='aa' * 8) if r['tap'] == 'dich' else r for r in _all(tmp_path, _rows())]
    o3 = O.build([dict(r, cham=r.get('cham_id', '')) for r in rows_d2], 'd2', 'cho', KHOA, CHO)  # D2: target runs
    assert o3['o'][0]['tap'] == 'dich' and o3['o'][0]['n_id'] == 3


def test_cli(tmp_path):
    p = tmp_path / 'rows.csv'
    rows = _rows() + _rows(n=3, tap='khop', luot='c1') + _rows(n=3, tap='dich', luot='r1')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(O.COLS) + ['controller', 'cham_id'], restval='')
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
        w = csv.DictWriter(f, fieldnames=list(O.COLS) + ['controller', 'cham_id'], restval='')
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
    dich = [dict(r, cham_id='aa' * 8) for r in dich]
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
    assert m['khung_mo'] == [['r1', 100 + i] for i in range(12)]              # the frames of cell A only
    # the A6 held-out check on the same kind of cells is still strict
    rk = _parse(_rows(n=3, tap='khop', luot='c1') + _rows(n=12, tap='kiem', luot='c2')
                + _rows(n=12, tap='kiem', luot='c2', x=0.7, d=0.006, s0=200) + _rows(n=3, tap='dich', luot='r1'),
                tmp_path, 'rk.csv')
    ok = O.build(rk, 'giu_ngoai_a6', 'cho', KHOA, CHO)
    assert ok['trang_thai'] == 'truot' and 'mat_na' not in ok
    assert O.build(_parse(_rows(n=3, tap='khop', luot='c1') + _rows(n=3, tap='kiem', luot='c2'), tmp_path, 'e.csv'),
                   'd2', 'cho', KHOA, CHO)['trang_thai'] == 'thieu'           # no target run, no cell


# ── Codex D2 round 1 (24/11 khuya): the cells on the deciding frames, every deciding node compares (D2-8) ─────────

def _parse_mh(rows, tmp_path, name='rows_mh.csv'):
    p = tmp_path / name
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(O.COLS) + ['mien_hop', 'controller', 'cham_id'], restval='')
        w.writeheader()
        w.writerows(rows)
    return O.read_rows(str(p))


def _dich(mh=None, **kw):
    """Target rows with mien_hop 1 everywhere; mh {(i, node): '0' | '1'} overrides."""
    kw.setdefault('s0', 100)
    rows = _rows(tap='dich', luot='r1', **kw)
    for r in rows:
        r['mien_hop'] = (mh or {}).get((int(r['id']) - kw['s0'], r['nut']), '1')
        r['cham_id'] = 'aa' * 8                       # every target run of a d2 file has a scoring id (V3-6)
    return rows


def _d2(tmp_path, dich, name='rows_mh.csv'):
    rows = _parse_mh(dich + _rows(n=3, tap='khop', luot='c1') + _rows(n=3, tap='kiem', luot='c2'), tmp_path, name)
    return O.build(rows, 'd2', 'cho', KHOA, CHO)


def test_d2_counts_only_frames_that_can_decide(tmp_path):
    # ten ids, one of them in no deciding domain at any node: nine ids, the cell is short of samples
    o = _d2(tmp_path, _dich(n=10, mh={(9, '0'): '0', (9, '1'): '0'}))
    c = o['o'][0]
    assert c['n_id'] == 9 and c['n_tho'] == 9 and not c['dat']
    m = o['mat_na']
    assert m['o_dat'] == [] and m['khung_mo'] == [] and m['n_ngoai_mien_quyet'] == 1 and m['ngoai_dai_u'] == []
    assert m['o_thieu'][0]['ly_do'] == ['thieu_mau']
    # ten deciding ids: dat; the id that decides nowhere is not in the mask either
    o = _d2(tmp_path, _dich(n=11, mh={(10, '0'): '0', (10, '1'): '0'}), 'b.csv')
    assert o['o'][0]['dat'] and o['o'][0]['n_id'] == 10
    assert o['mat_na']['khung_mo'] == [['r1', 100 + i] for i in range(10)]


def test_d2_every_deciding_node_must_compare(tmp_path):
    # id 3 has no comparison at node 1: where it cannot decide there, it is still comparable; where it can, thieu
    o = _d2(tmp_path, _dich(n=12, missing={(3, '1')}, mh={(3, '1'): '0'}))
    assert o['o'][0]['dat'] and o['o'][0]['n_id'] == 12 and o['o'][0]['n_thieu'] == 0
    o = _d2(tmp_path, _dich(n=12, missing={(3, '1')}), 'b.csv')
    c = o['o'][0]
    assert not c['dat'] and c['n_id'] == 11 and c['n_thieu'] == 1 and c['thieu_ids'] == [['r1', 103]]
    assert o['mat_na']['o_thieu'][0]['ly_do'] == ['thieu_id'] and o['mat_na']['khung_mo'] == []
    # the A6 held-out rows have no mien_hop column: every node counts, as before
    r = _all(tmp_path, _rows(missing={(3, '1')}))
    assert O.build(r, 'giu_ngoai_a6', 'cho', KHOA, CHO)['o'][0]['n_thieu'] == 1


def test_d2_worst_node_over_every_comparison(tmp_path):
    # id 5 is over the bound at node 1, where it cannot decide: the comparison still counts (the largest |d|)
    rows = _dich(n=12, mh={(5, '1'): '0'})
    for r in rows:
        if r['id'] == '105' and r['nut'] == '1':
            r['d_rad'] = 0.006
    o = _d2(tmp_path, rows)
    c = o['o'][0]
    assert not c['dat'] and c['ty_le_max'] == pytest.approx(1.2) and c['id_xau'] == ['r1', 105, '1']
    assert o['mat_na']['o_thieu'][0]['ly_do'] == ['vuot']


def test_d2_scoring_id_of_each_target_run(tmp_path):
    """Codex round 2 D2v2-4 (CODE 25/11): mat_na.lan_cham = the cham_id of the rows of each target run; a run with
    rows of two scorings (or with some rows without the id) is refused; rows without the column give no id."""
    def d2(dich, name):
        p = tmp_path / name
        with open(p, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(O.COLS) + ['mien_hop', 'controller', 'cham_id'], restval='')
            w.writeheader()
            w.writerows(dich + [dict(r, cham_id='dd' * 8) for r in _rows(n=3, tap='khop', luot='c1')]
                        + _rows(n=3, tap='kiem', luot='c2'))             # an id on a run outside the target set
        return O.build(O.read_rows(str(p)), 'd2', 'cho', KHOA, CHO)
    r1 =[dict(r, cham_id='aa' * 8) for r in _dich(n=12)]
    r2 = [dict(r, luot='r2', cham_id='bb' * 8) for r in _dich(n=12, x=0.7)]
    o = d2(r1 + r2, 'a.csv')
    assert o['mat_na']['lan_cham'] == {'r1': 'aa' * 8, 'r2': 'bb' * 8}
    assert len(o['mat_na']['khung_mo']) == 24                           # the cells do not depend on the id
    # Codex round 3 V3-6: a target run without an id is refused (its rows would fill cells shared with r1, r2):
    # nine ids of r1 in a cell + one id of r3 without a scoring id would open the cell for r1
    r3 = [dict(r, luot='r3', cham_id='') for r in _dich(n=1, s0=900)]
    with pytest.raises(O.OError, match='without cham_id'):
        d2(r1[:18] + r3, 'a3.csv')
    assert d2(r1[:18], 'a9.csv')['mat_na']['khung_mo'] == []            # nine ids alone: the cell stays shut
    A._check_sha(o, 'o')
    r1[5]['cham_id'] = 'cc' * 8                                         # one row of another scoring
    with pytest.raises(O.OError, match='more than one scoring'):
        d2(r1, 'b.csv')
    r1[5]['cham_id'] = ''                                               # one row without the id
    with pytest.raises(O.OError, match='more than one scoring'):
        d2(r1, 'c.csv')
    with pytest.raises(O.OError, match='without cham_id'):              # a d2 file from rows without the column
        _d2(tmp_path, [{k: v for k, v in r.items() if k != 'cham_id'} for r in _dich(n=12)], 'd.csv')


def test_d2_mien_hop_values_and_bands_of_the_code(tmp_path):
    rows = _dich(n=12)
    rows[0]['mien_hop'] = 'x'
    with pytest.raises(O.OError, match='mien_hop'):
        _d2(tmp_path, rows)
    # D2-15: the D2 bands are those of the code, not of the command line
    p = tmp_path / 'r.csv'
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(O.COLS) + ['mien_hop', 'controller', 'cham_id'], restval='')
        w.writeheader()
        w.writerows(_dich(n=12) + _rows(n=3, tap='khop', luot='c1') + _rows(n=3, tap='kiem', luot='c2'))
    k = tmp_path / 'khoa.json'
    k.write_text(json.dumps(LOCK), encoding='utf-8')
    base = ['--rows', str(p), '--muc-dich', 'd2', '--giu-ngoai-theo', 'cho', '--khoa-truoc', str(k)]
    assert O.main(base + ['--dai-u', '0,0.4,1.5', '--out', str(tmp_path / 'o1.json')]) == 2
    assert O.main(base + ['--dai-u', '0,0.3,0.6,1.0,1.5', '--out', str(tmp_path / 'o2.json')]) == 0
    assert O.main(base + ['--out', str(tmp_path / 'o3.json')]) == 0
    assert O.main(['--rows', str(p), '--muc-dich', 'giu_ngoai_a6', '--giu-ngoai-theo', 'cho', '--khoa-truoc', str(k),
                   '--dai-u', '0,0.4,1.5', '--out', str(tmp_path / 'o4.json')]) == 0     # the A6 check may set them
