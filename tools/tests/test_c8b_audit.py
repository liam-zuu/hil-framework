"""c8b_audit (design C8B v7 11b): counts, every hole in the decision domain, the upper-bound rule."""
import csv
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import c8b_audit as AU  # noqa: E402

META = dict(luot='C8-P-F08-1', bai='O1-F08', controller='PID')


def _frames(bad=(), why='xe_meo', gap_at=None, n=100):
    F = []
    t = 0.0
    for k in range(n):
        t += 1 / 60 if gap_at != k else 0.2
        F.append(dict(idx=k, t=t, valid=k not in bad, why='' if k not in bad else why, speed=0.5, wz=1.0))
    return F


def test_counts_and_holes():
    F = _frames(bad=range(10, 14), gap_at=50)
    o = AU.audit(F, 0.1445, META)
    key = 'xe_meo|u1|ccw|O1-F08|PID'                                       # u = 0.5 + 1.0 * 0.1445 / sqrt 2 = 0.602
    assert AU.u_of(F[0], 0.1445) == pytest.approx(0.5 + 0.1445 / 2 ** 0.5)
    assert AU.band(0.602) == 'u2' and o['dem'] == {'xe_meo|u2|ccw|O1-F08|PID': 4} and key not in o['dem']
    kinds = [(h['loai'], h['n_khung']) for h in o['ho']]
    assert kinds == [('khung_bi_loai', 4), ('ho_thoi_gian', 0)]
    h = o['ho'][0]
    assert h['t_a'] == F[9]['t'] and h['t_b'] == F[14]['t'] and h['ly_do'] == ['xe_meo']


def _gt(F, v):
    return {'G2': {f['t']: v for f in F if f['valid']}}


def test_upper_bound_rule_every_hole_in_the_domain():
    F = _frames(bad=range(10, 14), gap_at=50)
    dom = {'G2': [[0.0, 10.0]]}
    toc = {'G2': [[0.0, 1.5, 0.01]]}                                       # 10 mm/s
    # both holes in the domain, values 1 mm, threshold 5 mm: (1 + 1 + 0.01 dt)/2 < 5 -> excluded, finite
    o = AU.audit(F, 0.1445, META, dom, _gt(F, 0.001), toc, {'G2': 0.005})
    g = o['theo_G']['G2']
    assert g['n_ho'] == 2 and g['can_tren'] == 'huu_han' and all(h['ket_luan'] == 'loai_duoc' for h in g['ho'])
    v = g['ho'][1]
    assert v['gia_tri_kha_di_max'] == pytest.approx((0.001 + 0.001 + 0.01 * v['dt']) / 2)
    # a hole far from the largest value but able to decide: still thieu (V5-07)
    o2 = AU.audit(F, 0.1445, META, dom, _gt(F, 0.00499), toc, {'G2': 0.005})
    assert o2['theo_G']['G2']['can_tren'] == 'thieu'
    # no motion index -> thieu, never finite
    o3 = AU.audit(F, 0.1445, META, dom, None, None, None)
    assert o3['theo_G']['G2']['can_tren'] == 'thieu'
    # a hole outside the domain does not count
    o4 = AU.audit(F, 0.1445, META, {'G2': [[F[20]['t'], F[40]['t']]]}, None, None, None)
    assert o4['theo_G']['G2']['n_ho'] == 0 and o4['theo_G']['G2']['can_tren'] == 'huu_han'
    # a hole at the clip end in the domain: thieu
    o5 = AU.audit(_frames(bad=range(95, 100)), 0.1445, META, {'G2': [[0.0, 10.0]]}, None, None, None)
    assert o5['theo_G']['G2']['can_tren'] == 'thieu' and o5['theo_G']['G2']['ho'][0]['ket_luan'] == 'ho_o_dau_clip'


def test_rate_from_both_ends_and_unknown_u():
    F = _frames(bad=range(10, 14))
    F[14]['speed'] = 2.0                                                  # u at the far end outside every band
    o = AU.audit(F, 0.1445, META, {'G2': [[0.0, 10.0]]}, _gt(F, 0.001), {'G2': [[0.0, 1.5, 0.01]]}, {'G2': 0.005})
    assert o['theo_G']['G2']['can_tren'] == 'thieu' and o['theo_G']['G2']['ho'][0]['ket_luan'] == 'thieu_gia_tri_canh'


def test_cli(tmp_path):
    p = tmp_path / 'frames.csv'
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['idx', 't_s', 'valid', 'why', 'speed', 'wz'])
        for fr in _frames(bad=(5, 6)):
            w.writerow([fr['idx'], f'{fr["t"]:.6f}', int(fr['valid']), fr['why'], fr['speed'], fr['wz']])
    m = tmp_path / 'mien.json'
    m.write_text(json.dumps({'G2': [[0.0, 10.0]]}), encoding='utf-8')
    out = tmp_path / 'a.json'
    assert AU.main(['--frames', str(p), '--luot', 'r', '--bai', 'O1-F08', '--controller', 'PID', '--canh-m', '0.1445',
                    '--mien', str(m), '--out', str(out)]) == 0
    d = json.loads(out.read_text(encoding='utf-8'))
    assert d['version'] == 'c8b_audit_v1' and d['n_bi_loai'] == 2 and d['theo_G']['G2']['can_tren'] == 'thieu'
    m.write_text(json.dumps({'G2': [[3.0, 1.0]]}), encoding='utf-8')
    assert AU.main(['--frames', str(p), '--luot', 'r', '--bai', 'O1-F08', '--controller', 'PID', '--canh-m', '0.1445',
                    '--mien', str(m), '--out', str(tmp_path / 'b.json')]) == 2
