"""c8b_khoa_v5 (design C8B v7 sections 0.2, 5a, 5c, 6c, 10; J v6 section 6; L131, L132)."""
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import c8b_khoa_v5 as K  # noqa: E402

REAL = os.path.join('D:/CLAUDE_THESIS/05_DATA/results/C8/2026-11-18_run01/C1/c8b_3b_v3_ga')


def _tb(level, dung=True, p=0.002213, s=0.519, troi=True, alpha=False):
    return dict(version='c8b_3b_v2', task_rates=level, dung_duoc=dung, ga_ok=False, b_dung_ke_ca_troi=troi,
                d_p_quay_m=p if dung else None, d_psi_quay_deg=s if dung else None, thieu=[] if dung else ['B phu'],
                gioi_han=['x'], B=dict(shutter=dict(use_alpha=alpha, alpha_r_le_s_px=5e-6, alpha_c_le_s_px=-1e-6,
                                                     mot_khung_s=1 / 60, why='' if alpha else 'spread')))


def test_margin_once_and_limits():
    k = K.build({'O1-F05': _tb('O1-F05')})
    cq = k['muc']['O1-F05']['can_quay']
    assert cq['p_m'] == pytest.approx(1.25 * 0.002213) and cq['psi_deg'] == pytest.approx(1.25 * 0.519)
    assert cq['trang_thai'] == 'gioi_han' and cq['gioi_han'][0].startswith('L131') and cq['gioi_han'][1].startswith('L132')


def test_level_without_turning_bound_is_null_with_the_consequence():
    k = K.build({'O1-F08': _tb('O1-F08', dung=False)})
    cq = k['muc']['O1-F08']['can_quay']
    assert cq['p_m'] is None and cq['trang_thai'] == 'thieu' and '12(b)' in cq['he_qua']
    assert k['muc']['O1-F02']['can_quay']['trang_thai'] == 'thieu'          # no file at all: thieu, not 0
    assert k['muc']['O1-F08']['chi_bao_cao'] == ['g5'] and k['muc']['O1-F05']['chi_bao_cao'] == []


def test_shutter_branch_per_level():
    ng = [dict(vai='3b O1-F02', file='f02.json', duong_dan='D:/x/f02.json', sha256='cd' * 32)]
    k = K.build({'O1-F02': _tb('O1-F02', alpha=True), 'O1-F05': _tb('O1-F05')}, nguon=ng)
    assert k['muc']['O1-F02']['man_trap']['nhanh'] == 'alpha' and k['muc']['O1-F02']['man_trap']['alpha_c_s_px'] == 1e-6
    mt = k['muc']['O1-F05']['man_trap']
    assert mt['nhanh'] == 'mot_khung' and mt['le'] is None and mt['mot_khung_s'] == pytest.approx(1 / 60)


def test_level_mismatch_refused():
    with pytest.raises(K.KhoaError):
        K.build({'O1-F05': _tb('O1-F02')})


def test_u_beta_cam_margin_and_missing():
    ll = dict(beta_cam_replay=dict(goc=dict(max_abs_dbeta_rad=math.radians(0.1), trang_thai='dat'),
                                   K1=dict(max_abs_dbeta_rad=math.radians(0.2), trang_thai='dat')))
    k = K.build({}, ll)
    assert k['u_beta_cam_deg']['goc'] == pytest.approx(0.125) and k['u_beta_cam_deg']['K2'] is None
    assert k['u_beta_cam_trang_thai'] == 'thieu' and any('K2' in w for w in k['u_beta_cam_ly_do'])
    assert K.build({})['u_beta_cam_trang_thai'] == 'thieu'


@pytest.mark.skipif(not os.path.isdir(REAL), reason='session-1 3b results not on this machine')
def test_cli_on_session1_l132(tmp_path, capsys):
    args = []
    for lv in ('F02', 'F05', 'F08'):
        args += ['--tb', f'O1-{lv}={REAL}/c8b_3b_A2_B_ea0ec4e_O1-{lv}_L132.json']
    out = tmp_path / 'k.json'
    assert K.main(args + ['--out', str(out)]) == 0
    d = json.loads(out.read_text(encoding='utf-8'))
    assert d['muc']['O1-F05']['can_quay']['p_m'] == pytest.approx(1.25 * 0.0022129880449652806)
    assert d['muc']['O1-F08']['can_quay']['p_m'] is None
    assert len(d['khoa_truoc']['nguon']) == 3 and all(n['sha256'] for n in d['khoa_truoc']['nguon'])


def test_bang_man_trap_has_k_and_alpha_source():
    ng = [dict(vai='3b O1-F05', file='f05.json', duong_dan='D:/x/f05.json', sha256='ab' * 32)]
    k = K.build({'O1-F05': _tb('O1-F05', alpha=True), 'O1-F02': _tb('O1-F02')}, nguon=ng)
    b = k['bang_man_trap']
    assert set(b) == {'O1-F02', 'O1-F05', 'O1-F08', 'K'} and b['K']['nhanh'] == 'alpha'
    assert b['K']['nguon_3b']['sha256'] == 'ab' * 32 and b['O1-F02'] == dict(nhanh='mot_khung')
    with pytest.raises(K.KhoaError):
        K.build({'O1-F05': _tb('O1-F05', alpha=True)})               # alpha without its source


def test_m5b_bands_closed_terms_branch_and_state(tmp_path):
    # M5b (GATE 8705918 GC6-5, names CODE 19/11): one band [0, v_max + wz_max a / sqrt 2] per level with P4 / A4 after
    # the margin; thieu -> null values; hang_dong keys exactly the gate's, null until their sources; nhanh_tick from
    # the plan; trang_thai khoa only when every term of every level of muc_kha_thi is there
    from o1_feasibility import Schedule
    from o1_tasks import FIG8
    import o1_gate
    a = 0.0964
    k = K.build({'O1-F05': _tb('O1-F05'), 'O1-F08': _tb('O1-F08', dung=False)}, a_m=a, nhanh_tick='bo_tick')
    v, wz = o1_gate.fig8_cmd_max(Schedule(FIG8['O1-F05']))
    b = k['muc']['O1-F05']['hang_theo_u']
    assert len(b) == 1 and b[0]['u_lo_m_s'] == 0.0 and b[0]['u_hi_m_s'] == pytest.approx(v + wz * a / math.sqrt(2))
    assert b[0]['P4_m'] == pytest.approx(1.25 * 0.002213) and b[0]['A4_deg'] == pytest.approx(1.25 * 0.519)
    assert b[0]['trang_thai'] == 'gioi_han'
    b8 = k['muc']['O1-F08']['hang_theo_u'][0]
    assert b8['trang_thai'] == 'thieu' and b8['P4_m'] is None and b8['A4_deg'] is None
    hd = k['muc']['O1-F05']['hang_dong']
    assert set(hd) == {'a_m', 'hang_m', 'he_so_r', 'u_phi_rad', 'hang_rad'} and hd['a_m'] == a
    assert all(hd[q] is None and k['muc']['O1-F05']['hang_dong_ly_do'][q] for q in ('hang_m', 'he_so_r', 'u_phi_rad',
                                                                                   'hang_rad'))
    assert k['nhanh_tick'] == 'bo_tick' and k['trang_thai'] == 'chua_khoa' and 'hang_dong' in k['ly_do_chua_khoa']
    assert 'ke hoach' in k['ly_do_chua_khoa']
    # every term there (the other sessions' sources filled in by hand here): khoa
    ng = [dict(vai='ke_hoach_c8', file='kh.json', sha256='cd' * 32)]
    k2 = K.build({'O1-F05': _tb('O1-F05')}, nguon=ng, a_m=a, nhanh_tick='giu_tick', muc_kha_thi=['O1-F05'])
    k2['muc']['O1-F05']['hang_dong'].update(hang_m=0.004, he_so_r=0.002, u_phi_rad=0.001, hang_rad=0.01)
    assert K.trang_thai_khoa(k2, ['O1-F05']) == ('khoa', None)
    # no tag side: no band, not lockable
    k3 = K.build({'O1-F05': _tb('O1-F05')})
    assert k3['muc']['O1-F05']['hang_theo_u'] == [] and k3['muc']['O1-F05']['hang_theo_u_ly_do']


def test_m5b_cli_plan_branch(tmp_path):
    p = tmp_path / '3b.json'
    p.write_text(json.dumps(_tb('O1-F05')), encoding='utf-8')
    kh = tmp_path / 'ke_hoach_c8.json'
    kh.write_text(json.dumps(dict(format='ke_hoach_c8_v1', nhanh_tick='bo_tick', o=[])), encoding='utf-8')
    base = ['--tb', f'O1-F05={p}', '--canh-ma-m', '0.0964', '--muc', 'O1-F05']
    out = tmp_path / 'k.json'
    assert K.main(base + ['--ke-hoach', str(kh), '--out', str(out)]) == 0
    d = json.loads(out.read_text(encoding='utf-8'))
    assert d['nhanh_tick'] == 'bo_tick' and d['muc_kha_thi'] == ['O1-F05']
    assert any(n['vai'] == 'ke_hoach_c8' and len(n['sha256']) == 64 for n in d['khoa_truoc']['nguon'])
    assert K.main(base + ['--ke-hoach', str(kh), '--nhanh-tick', 'giu_tick', '--out', str(tmp_path / 'k2.json')]) == 2
    assert K.main(base + ['--muc', 'O1-F99', '--out', str(tmp_path / 'k3.json')]) == 2
