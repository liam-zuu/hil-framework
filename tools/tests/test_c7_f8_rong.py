"""F8 rong, cach A (Liam 23/11, phien CODE local_483c9ce2, hop chon "phi buoi 2", nguyen van: "Cach A (Recommended)";
CODE 03_CODE/docs/2026-11-23_Chot_T12_khoa_C8B_v1.md muc 4): an F8 that passes its identity checks but gives no run a
lan_lap_id -> beta = 0, u_beta = 0.234 deg (A2 of session 1), plate khong_doi, labelled; anything else keeps the old
rule (beta 'dat' and the 12b binding through lan_lap_v1)."""
import json
import math
import os

import pytest

import c7_chuoi_khung as ck
import c7_score as cs
import c7_tuong_doi as ctd

A6 = 'a' * 64
CLIP12 = 'c' * 64
R2 = 'D:/CLAUDE_THESIS/05_DATA/results/C8/2026-11-20_run02'


def run_of(**over):
    """One run of an F8 rong file as doc_lan_lap returns it (session 2 shape: beta measured, not validated)."""
    d = dict(schema='lan_lap_v1', luot_id='C8-P-F05-1', cho='goc', tam_id='tam_c8_1', beta_deg=-0.559,
             u_beta_deg=None, tam_doi_trong_luot=None, cung_lan_lap=None, lan_lap_id=None, chuan_12b=None,
             lan_lap_id_ly_do='lan_lap_khong_xac_dinh',
             trang_thai={'beta': 'chua_ket_luan', 'delta_c_B': 'chua_ket_luan'},
             a6={'file': 'a6.json', 'sha256': A6, 'tam_id': 'tam_c8_1'}, _moi_lan_lap_id_null=True,
             _sha256='f' * 64, _path='lan_lap.json')
    d.update(over)
    return d


def tx(**over):
    d = dict(schema='truc_xe_v1', mien_hieu_luc=dict(cho='goc', thu_id_cap='HC_goc_2', lan_lap_id=None),
             clip_sha256=CLIP12, trang_thai='khoa', phi_hat_chuan_k_deg=1.5, u_phi_hat_chuan_k_deg=0.04,
             psi_chuan_k_deg=44.0, psi_chuan_k_nguon='ref_12b_camera_khong_F6')
    d.update(over)
    return d


def test_f8_rong_holds_with_every_identity_check():
    ok, why = ctd.f8_rong(run_of(), A6, True, tx())
    assert ok and why == []
    assert ctd.f8_rong(run_of(), A6, True, None)[0]                 # no truc_xe: the F8 side alone


@pytest.mark.parametrize('ll, a6, ref, t, frag', [
    (run_of(lan_lap_id='L0'), A6, True, tx(), 'co lan_lap_id'),
    (run_of(_moi_lan_lap_id_null=False), A6, True, tx(), 'co lan_lap_id'),
    (run_of(tam_id=None), A6, True, tx(), 'tam_id'),
    (run_of(tam_id=''), A6, True, tx(), 'tam_id'),
    (run_of(), 'b' * 64, True, tx(), 'a6.sha256'),
    (run_of(), None, True, tx(), 'a6.sha256'),
    (run_of(a6=None), A6, True, tx(), 'a6.sha256'),
    (run_of(), A6, False, tx(), 'ref_12b'),
    (run_of(), A6, None, tx(), 'ref_12b'),
    (run_of(), A6, True, tx(mien_hieu_luc=dict(cho='goc', thu_id_cap='HC_goc_2', lan_lap_id='L0')), 'truc_xe_v1 co'),
    (run_of(), A6, True, tx(mien_hieu_luc=dict(cho='goc', lan_lap_id=None, lan_lap_v1_sha256='e' * 64)),
     'lan_lap_v1_sha256'),
])
def test_f8_rong_refused_on_each_single_failure(ll, a6, ref, t, frag):
    ok, why = ctd.f8_rong(ll, a6, ref, t)
    assert not ok and any(frag in w for w in why), why


def test_f8_rong_none_lan_lap():
    assert ctd.f8_rong(None, A6, True) == (False, ['thieu_lan_lap_v1'])


def _tcp(psi_deg=45.3, **over):
    d = dict(tap=ctd.TAP_MA_CACH_A, psi_ok=True, psi_tc_rad=math.radians(psi_deg))
    d.update(over)
    return d


def test_phi_beta_cach_a_beta_is_the_body_turn_u_beta_0234():
    """CODE erratum 23/11 (goc 77991725): beta = wrap(psi'_tc - psi_chuan_k) (the body turns as the tag), not 0."""
    t = tx()
    phi, u, ok, why = ctd.phi_beta('O1-F05', t, run_of(), _tcp(), 0.02, f8_rong_ok=True)
    assert ok and why == []
    assert phi == pytest.approx(math.radians(1.5 + (45.3 - 44.0)), abs=1e-12)   # not the run's beta -0.559, not 0
    assert u == pytest.approx(math.radians(1.25 * (0.04 + 0.234) + 0.02), abs=1e-15)
    phi, u, ok, why = ctd.phi_beta('O1-F05', t, run_of(), _tcp(), None, f8_rong_ok=True)
    assert ok and math.isinf(u)                                        # u_beta_cam still from the C8B lock
    t2 = tx(psi_chuan_k_deg=179.0)
    phi, _, ok, _ = ctd.phi_beta('O1-F05', t2, run_of(), _tcp(-179.5), 0.0, f8_rong_ok=True)
    assert ok and phi == pytest.approx(math.radians(1.5 + 1.5), abs=1e-12)      # wrapped across pi


@pytest.mark.parametrize('t, tc, frag', [
    (tx(psi_chuan_k_deg=None), _tcp(), 'psi_chuan_k_deg'),
    (tx(psi_chuan_k_deg=float('nan')), _tcp(), 'psi_chuan_k_deg'),
    (tx(), _tcp(psi_tc_rad=None), 'psi_tc_rad'),
    (tx(), _tcp(psi_ok=False), 'psi_tc_rad'),
])
def test_phi_beta_cach_a_needs_both_headings(t, tc, frag):
    phi, u, ok, why = ctd.phi_beta('O1-F05', t, run_of(), tc, 0.0, f8_rong_ok=True)
    assert not ok and phi is None and math.isinf(u) and any(frag in w for w in why), why


def test_phi_beta_cach_a_keeps_the_anchor_rule_and_truc_xe_checks():
    for tap in ('ma_mat_neo', 'giao'):            # CODE 6812dcf6: W of cach A is the tag-gate set, nothing else
        _, _, ok, why = ctd.phi_beta('O1-F05', tx(), run_of(), _tcp(tap=tap), 0.0, f8_rong_ok=True)
        assert not ok and any('khong theo cong ma' in w for w in why)
    _, _, ok, why = ctd.phi_beta('O1-F05', tx(trang_thai='tam'), run_of(), _tcp(), 0.0, f8_rong_ok=True)
    assert not ok and any('khong khoa' in w for w in why)
    with pytest.raises(ValueError, match='place'):
        ctd.phi_beta('O1-K1', tx(), run_of(), _tcp(), 0.0, f8_rong_ok=True)


def test_phi_beta_without_the_branch_is_the_old_rule():
    _, u, ok, why = ctd.phi_beta('O1-F05', tx(), run_of(), _tcp(tap='giao'), 0.0)
    assert not ok and math.isinf(u)
    assert any('trang_thai.beta' in w for w in why) and any('lan_lap_id' in w for w in why)


def test_tam_doi_of_cach_a():
    assert cs.tam_doi_of(run_of(), f8_rong_ok=True) == 'khong_doi'
    assert cs.tam_doi_of(run_of()) == 'khong_xac_dinh'
    assert cs.tam_doi_of(None, f8_rong_ok=True) == 'khong_xac_dinh'
    assert cs.tam_doi_of(run_of(tam_doi_trong_luot=True)) == 'doi'


def _tc(**over):
    d = dict(tap=ctd.TAP_MA_CACH_A, p_ok=True, psi_ok=True, trang_thai='dat', psi_tc_rad=math.radians(45.3))
    d.update(over)
    return d


def test_build_rel_records_the_label_and_gives_the_position_basis():
    rel = cs.build_rel('O1-F05', [], _tc(), tx(), run_of(), 0.0, None, a6_sha256=A6, co_ref_12b=True)
    assert rel['can_cu'] and rel['phi_ok'] and rel['tam_doi'] == 'khong_doi'
    f8 = rel['f8_rong']
    assert f8['ok'] and f8['nhan'] == 'can_lech_tam_buoi1_chua_xac_minh_lan_lap'
    assert f8['beta_deg'] == pytest.approx(45.3 - 44.0, abs=1e-12) and f8['u_beta_deg'] == 0.234 and f8['gia_dinh']
    assert rel['phi'] == pytest.approx(math.radians(1.5 + 1.3), abs=1e-12)


def test_build_rel_without_identity_stays_t_som():
    rel = cs.build_rel('O1-F05', [], _tc(), tx(), run_of(), 0.0, None, a6_sha256='b' * 64, co_ref_12b=True)
    assert not rel['can_cu'] and not rel['phi_ok'] and rel['tam_doi'] == 'khong_xac_dinh'
    assert not rel['f8_rong']['ok'] and rel['f8_rong']['ly_do']
    rel = cs.build_rel('O1-F05', [], _tc(), tx(), run_of())          # old callers: no a6 / ref_12b -> no branch
    assert not rel['can_cu'] and not rel['f8_rong']['ok']


def _write(tmp_path, runs):
    p = tmp_path / 'lan_lap.json'
    p.write_text(json.dumps(dict(schema='lan_lap_v1', tam_id='tam_c8_1', a6={'sha256': A6}, luot=runs)),
                 encoding='utf-8')
    return str(p)


def test_doc_lan_lap_flag_is_over_the_whole_file(tmp_path):
    a = dict(schema='lan_lap_v1', luot_id='r1', cho='goc', lan_lap_id=None)
    b = dict(schema='lan_lap_v1', luot_id='r2', cho='goc', lan_lap_id=None)
    assert ck.doc_lan_lap(_write(tmp_path, [a, b]), 'r1')['_moi_lan_lap_id_null'] is True
    b2 = dict(b, lan_lap_id='L0')
    r1 = ck.doc_lan_lap(_write(tmp_path, [a, b2]), 'r1')
    assert r1['_moi_lan_lap_id_null'] is False and r1['lan_lap_id'] is None    # another run has an id: not F8 rong
    assert ctd.f8_rong(r1, A6, True)[0] is False
    r1 = ck.doc_lan_lap(_write(tmp_path, [a, b]), 'r1')
    assert r1['tam_id'] == 'tam_c8_1' and r1['a6']['sha256'] == A6                # file keys copied in
    assert ctd.f8_rong(r1, A6, True)[0] is True


@pytest.mark.skipif(not os.path.exists(R2 + '/nhac_tam_v4/ll1_bang_dinh/lan_lap.json'), reason='session 2 data')
def test_session_2_f8_is_the_f8_rong_case():
    """F8 of session 2 (sha256 5b4238d8, gitroot 5dfd9b33), A6 5f6a1fdd, F2 of session 2: every run is the cach A case."""
    f8 = R2 + '/nhac_tam_v4/ll1_bang_dinh/lan_lap.json'
    cua = ctd.read_jsonl(R2 + '/F2/cua_so_v1.jsonl')
    a6 = ck.doc_a6(R2 + '/c8b/T7_a6_rho/c8b_a6_v2_rho.json')['sha256']
    runs = json.load(open(f8, encoding='utf-8'))['luot']
    assert len(runs) == 14
    for r in runs:
        ll = ck.doc_lan_lap(f8, r['luot_id'])
        ref = any(x.get('loai') == 'ref_12b' and x.get('cho') == ll['cho'] for x in cua)
        ok, why = ctd.f8_rong(ll, a6, ref)
        assert ok, (r['luot_id'], why)


def test_co_ref_12b_is_of_the_place():
    rows = [dict(loai='pre', cho='goc'), dict(loai='ref_12b', cho='K1'), dict(loai='dung_12b', cho='goc')]
    assert cs.co_ref_12b(rows, 'K1') and not cs.co_ref_12b(rows, 'goc') and not cs.co_ref_12b([], 'goc')


def _chain(ids):
    return [dict(frame_id=i, t_cam=i / 60.0, tag_x=0.4, tag_y=-0.2, tag_psi=0.3, psi_ma_p=0.3, p_x=0.4, p_y=-0.2,
                 cong_ok=True, a6_ok=True, q_ok=True) for i in ids]


def _ll_neo(ids):
    return dict(dau={'pre': dict(khung_giao_ma_neo=dict(id=ids, pts_s=[i / 60.0 for i in ids]),
                                 khung_giao_sha256=ctd.ids_sha(ids), mat_neo=False)})


def test_tham_chieu_cach_a_w_is_the_tag_gate_set():
    """CODE 23/11 (chot T12 muc 4, goc 6812dcf6): in the F8 rong branch W = the tag-gate frames of pre, label
    w_theo_ma_cach_A; anchors lost in the middle (session 2, segment 3) no longer make a hole; outside the branch the
    intersection of J v6 2 is kept (and the hole loses the window)."""
    ma = list(range(180))
    pre = dict(khung_ma_hop_le=ma, t_ref=3.0, ket_luan=dict(dat=True, ly_do=[]))
    neo = [i for i in ma if not 60 <= i < 75]                         # 15 anchor frames lost: a 0.25 s hole
    kt = dict(ctd.KHUNG_TC_DE_XUAT, nguon='test')
    old = ctd.tham_chieu(pre, _chain(ma), _ll_neo(neo), kt, 0.10, 1 / 60.0)
    assert old['tap'] == 'giao' and old['n'] == len(neo) and old['trang_thai'] == 'hong_cua_so_tc'
    assert old['w_nhan'] is None
    new = ctd.tham_chieu(pre, _chain(ma), _ll_neo(neo), kt, 0.10, 1 / 60.0, f8_rong_ok=True)
    assert new['tap'] == ctd.TAP_MA_CACH_A and new['w_nhan'] == 'w_theo_ma_cach_A' and not new['mat_neo']
    assert new['ids'] == ma and new['n'] == 180 and new['trang_thai'] == 'dat'
    # the sample gate still holds on the tag set: a tag hole loses the window as before
    ma2 = [i for i in ma if not 60 <= i < 75]
    pre2 = dict(khung_ma_hop_le=ma2, t_ref=3.0, ket_luan=dict(dat=True, ly_do=[]))
    g = ctd.tham_chieu(pre2, _chain(ma2), None, kt, 0.10, 1 / 60.0, f8_rong_ok=True)
    assert g['trang_thai'] == 'hong_cua_so_tc' and g['tap'] == ctd.TAP_MA_CACH_A
