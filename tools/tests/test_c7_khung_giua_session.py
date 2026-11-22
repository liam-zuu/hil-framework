"""Tests of tools/c7_khung_giua_session.py (c7_khung_giua_session_v1 phys_v1 from the CODE_KHUNG estimator output) on a
SYNTHETIC estimator file (not a measurement); the files written are read back by the two readers (NHACTAM, HIEUCHUAN)."""
import copy
import hashlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import c7_khung_giua_session as kg  # noqa: E402
import c7_nhac_tam as nt  # noqa: E402
import c8_hc_a as hc  # noqa: E402

CHO = {'goc': [0.0, 0.0], 'K1': [1.0, 0.0], 'K2': [-0.5, -0.5]}
DOAN = {'1': ['H0a', 'H0b'], '2': ['H2a', 'H2b', 'Hm']}


def _sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def _entry(a, b, sa, sb, cho, base):
    k = dict(mo_hinh='phys', du_tinh_tien_m=base * 2, du_xoay_rad=base, du_thang=base / 2, du_bien_dang_rad=base * 1.2,
             xoay_bien_thien_rad=base / 10, thang_bien_thien=base / 20,
             hanh_lang=dict(du_xoay_rad=base, du_thang=base / 2, du_bien_dang_rad=base * 1.3),
             uoc_tai_cho_chinh=dict(omega=-base / 5, se_omega=base / 3, sigma=base / 7, se_sigma=base / 4,
                                    gamma=base / 6, se_gamma=base / 8, d=0.0, se_d=0.0))
    return dict(doan=['1', '2'], session=[a, b], session_sha256=[sa, sb], cho=cho, n_to_hop=2,
                to_hop=[f'{a}@{a} | {b}@{b}', f'{a}@{a} | {b}@Hm'], n_moc=32, du_m=4e-4, du_m_max_to_hop=5e-4, khung=k,
                phan_du=dict(s_du_m=1.5e-4, du_sau_khop_max_m=5e-4, loo_rms_m=1.6e-4, tuong_quan_du_lan_can=[-0.2, -0.1]),
                phat_hien=dict(delta_canh_tai_cho_m=7e-4, delta_canh_sd_du_m=2e-4, xoay_moc_sd_rad=2e-3))


@pytest.fixture
def ca(tmp_path):
    s = {}
    for n in ('H0a', 'H2a'):
        p = tmp_path / f'{n}.json'
        p.write_text(json.dumps(dict(kind='c7_session', ten=n)), encoding='utf-8')
        s[n] = (str(p), _sha(p))
    uoc = dict(schema='x', hil=dict(commit='05b994c'), doan=DOAN, cho=CHO, rho_m=0.4, k_se=3.0,
               session={n: dict(file=v[0], sha256=v[1]) for n, v in s.items()},
               cap=[dict(ten='H0a|H0b (cam H0a)', loai='trong_doan_chuoi', khop=dict(phys=dict(s_du_m=4.6e-5))),
                    dict(ten='H2a|H2b (cam H2a)', loai='trong_doan_chuoi', khop=dict(phys=dict(s_du_m=1.7e-5))),
                    dict(ten='H2a|Hm (cam H2a)', loai='trong_doan_chuoi', khop=dict(phys=dict(s_du_m=9e-5)))],
               tong_hop=[_entry('H0a', 'H2a', s['H0a'][1], s['H2a'][1], c, 1e-4 * (i + 1)) for i, c in enumerate(CHO)])
    script = tmp_path / 'uoc_khung_giua_session.py'
    script.write_text('# estimator\n', encoding='utf-8')
    return dict(d=tmp_path, s=s, uoc=uoc, script=str(script))


def run(c, uoc=None, out='ra'):
    p = c['d'] / 'uoc.json'
    p.write_text(json.dumps(uoc or c['uoc']), encoding='utf-8')
    argv = ['--uoc', str(p), '--uoc-script', c['script'], '--out-dir', str(c['d'] / out)]
    for n, v in c['s'].items():
        argv += ['--session', f'{n}={v[0]}']
    return kg.main(argv)


def test_ghi_file_ca_hai_ben_doc_duoc(ca):
    assert run(ca) == 0
    f = ca['d'] / 'ra' / 'khung_giua_session_H0a_H2a.json'
    o = json.load(open(f, encoding='utf-8'))
    assert o['schema'] == 'c7_khung_giua_session_v1' and o['mo_hinh'] == 'phys_v1' and o['truoc_le'] is True
    assert o['session_sha256'] == [ca['s']['H0a'][1], ca['s']['H2a'][1]] and o['du_m'] == 4e-4 and o['n_moc'] == 32
    assert o['theo_cho']['K1']['du_xoay_rad'] == pytest.approx(2e-4)
    assert o['theo_cho']['K2']['hanh_lang_12b']['du_bien_dang_rad'] == pytest.approx(3e-4 * 1.3)
    assert o['nen_nhieu'] == [dict(cap='H0a|H0b', s_du_m=4.6e-5), dict(cap='H2a|H2b', s_du_m=1.7e-5)]
    assert o['phat_hien_ma']['delta_canh_m']['goc'] == 7e-4
    assert {x['vai_tro'] for x in o['nguon']} == {'uoc', 'uoc_script', 'session_H0a', 'session_H2a'}
    k = nt.doc_kgs([str(f)])
    v = next(iter(k.values()))
    assert v['mo_hinh'] == 'phys_v1' and v['theo_cho']['goc']['du_bien_dang_rad'] == pytest.approx(1.2e-4)
    r = hc.read_khung_giua_session([str(f)])
    assert next(iter(r.values()))['hanh_lang_12b_du_bien_dang_rad']['K1'] == pytest.approx(2e-4 * 1.3)


def test_thieu_cho_bi_tu_choi(ca):
    u = copy.deepcopy(ca['uoc'])
    u['tong_hop'] = [t for t in u['tong_hop'] if t['cho'] != 'K2']
    assert run(ca, u) == 2
    assert not (ca['d'] / 'ra').exists()


def test_session_khac_file_bi_tu_choi(ca):
    u = copy.deepcopy(ca['uoc'])
    for t in u['tong_hop']:
        t['session_sha256'][1] = '0' * 64
    assert run(ca, u) == 2


@pytest.mark.parametrize('f', ['du_xoay_rad', 'du_bien_dang_rad'])
def test_so_am_hay_khong_huu_han_bi_tu_choi(ca, f):
    for bad in (-1e-5, float('nan')):
        u = copy.deepcopy(ca['uoc'])
        u['tong_hop'][1]['khung'][f] = bad
        assert run(ca, u, out=f'ra_{f}_{bad}') == 2


def test_khong_phai_phys_bi_tu_choi(ca):
    u = copy.deepcopy(ca['uoc'])
    u['tong_hop'][0]['khung']['mo_hinh'] = 'quad'
    assert run(ca, u) == 2


def test_khong_ghi_de(ca):
    assert run(ca) == 0
    assert run(ca) == 2
