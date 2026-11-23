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


def test_l136_f08_with_drift():
    # L136 (Liam 20/11 at CODE "Ap L132 cho F08 (Recommended)"): F08 from the DOCNHOE frames, same L132 formula
    # (x 1.25), the L136 limit written; G5 F08 still report only
    k = K.build({'O1-F08': _tb('O1-F08', p=0.002405, s=0.7088)})
    cq = k['muc']['O1-F08']['can_quay']
    assert cq['p_m'] == pytest.approx(1.25 * 0.002405) and cq['psi_deg'] == pytest.approx(1.25 * 0.7088)
    assert [g[:4] for g in cq['gioi_han'][:3]] == ['L131', 'L132', 'L136']
    assert k['muc']['O1-F08']['chi_bao_cao'] == ['g5']
    k5 = K.build({'O1-F05': _tb('O1-F05')})
    assert not any(g.startswith('L136') for g in k5['muc']['O1-F05']['can_quay']['gioi_han'])


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
    k2 = K.build({'O1-F05': _tb('O1-F05')}, nguon=ng, a_m=a, nhanh_tick='giu_tick', muc_kha_thi=['O1-F05'],
                 nguong=NG, a6=_a6(), ses=SES, cam=CAMF, lan_lap=_f8())
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


def _a6(dat=True, **lan):
    return dict(schema='c8b_a6_v2', tam_id='T1', h_goc_m=[0.1250, 0.1236, 0.1232, 0.1246], u_h_goc_m=[0.0012] * 4,
                trang_thai=dict(a6='thieu', h='dat' if dat else 'thieu'),
                lan_lap_cho=lan or dict(goc=dict(lan_lap_id='L0', luot_id_E='Eg'),
                                        K1=dict(lan_lap_id='L0', luot_id_E='Ek')))


def _f8(rows=None, tam='T1'):
    """F8 lan_lap_v1 (NHACTAM): one record per run with pre / post windows; the E clips usually have none."""
    return dict(schema='lan_lap_v1', tam_id=tam, luot=rows if rows is not None else [
        dict(luot_id='r1', cho='goc', lan_lap_id='L0'), dict(luot_id='r2', cho='goc', lan_lap_id=None),
        dict(luot_id='r3', cho='K1', lan_lap_id='L0')])


def test_a6c1_7_lan_lap_E_against_f8():
    # CODE 20/11 (data contract ~05:00): an F8 run of the E clip must agree; else every non-null F8 mounting of the
    # place one value equal to the declared one; else chua_ket_luan lan_lap_E_khong_doi_chieu
    r, st, why = K.lan_lap_E(_a6(), _f8())
    assert st == 'dat' and r['goc']['trang_thai'] == 'dat' and r['K1']['trang_thai'] == 'dat', why
    r, st, _ = K.lan_lap_E(_a6(), _f8(_f8()['luot'] + [dict(luot_id='Eg', cho='goc', lan_lap_id='L0')]))
    assert st == 'dat'
    with pytest.raises(K.KhoaError):                                       # the E run in F8 with another mounting
        K.lan_lap_E(_a6(), _f8(_f8()['luot'] + [dict(luot_id='Eg', cho='goc', lan_lap_id='L1')]))
    r, st, _ = K.lan_lap_E(_a6(), _f8(_f8()['luot'] + [dict(luot_id='Eg', cho='goc', lan_lap_id=None)]))
    assert st == 'chua_ket_luan' and r['goc']['trang_thai'] == 'chua_ket_luan'
    r, st, why = K.lan_lap_E(_a6(), _f8(_f8()['luot'] + [dict(luot_id='r4', cho='goc', lan_lap_id='L1')]))
    assert st == 'chua_ket_luan' and r['goc']['trang_thai'] == 'chua_ket_luan' and r['K1']['trang_thai'] == 'dat'
    assert K.LL_E_LY_DO in r['goc']['ly_do'] and any('goc' in w for w in why)
    r, st, _ = K.lan_lap_E(_a6(), _f8([dict(luot_id='r1', cho='goc', lan_lap_id='L5')]))   # one value, not declared
    assert r['goc']['trang_thai'] == 'chua_ket_luan' and r['K1']['trang_thai'] == 'chua_ket_luan'  # K1: no F8 value
    assert K.lan_lap_E(_a6(), None)[1] == 'chua_ket_luan'
    # CODE 23/11 (Chot_F8_rong_buoi2_v1 muc 2): every F8 id null -> no comparison, dat with the assumption label
    nul = _f8([dict(luot_id='r1', cho='goc', lan_lap_id=None), dict(luot_id='Ek', cho='K1', lan_lap_id=None)])
    r, st, why = K.lan_lap_E(_a6(), nul)
    assert st == 'dat' and not why and all(r[c]['nhan'] == K.LL_E_NULL and r[c]['gia_dinh'] for c in ('goc', 'K1'))
    assert K.lan_lap_E(_a6(goc=dict(lan_lap_id='L0', luot_id_E=None)), nul)[1] == 'chua_ket_luan'   # still declared
    assert K.lan_lap_E(_a6(), _f8([]))[1] == 'chua_ket_luan'                     # no run is not "all null"
    with pytest.raises(K.KhoaError):                                              # a plate mismatch still refused
        K.lan_lap_E(_a6(), _f8([dict(luot_id='r1', cho='goc', lan_lap_id=None)], tam='T9'))
    assert 'nhan' not in K.lan_lap_E(_a6(), _f8())[0]['goc']                      # an F8 with ids: rule unchanged
    assert K.lan_lap_E(dict(_a6(), lan_lap_cho=None), _f8())[1] == 'chua_ket_luan'
    assert K.lan_lap_E(_a6(goc=dict(lan_lap_id='L0', luot_id_E=None)), _f8())[1] == 'chua_ket_luan'
    with pytest.raises(K.KhoaError):
        K.lan_lap_E(_a6(), _f8(tam='T2'))
    k = K.build({'O1-F05': _tb('O1-F05')}, a6=_a6(), lan_lap=_f8([dict(luot_id='r4', cho='goc', lan_lap_id='L1')]))
    assert k['lan_lap_E_trang_thai'] == 'chua_ket_luan' and 'lan_lap_E' in k['ly_do_chua_khoa']


SES = dict(car_side_m=0.1445, tag_h_m=0.124175, kappa=0.124175 / 2.632, car_pose_version='car_pose_v2')
CAMF = dict(day_doi=dict(dzc_m=0.01, dn_m=0.016))
NG = dict(format='nguong_khoa_c8_v1', eps_H=dict(mm=2.0, huong_deg=0.5), quyet_liam='(gia, test)')


def test_eps_h_and_mien_car_pose(tmp_path):
    # CODE 20/11: eps_H from nguong_khoa_c8_v1; mien_car_pose h from the A6 corner heights, kappa derived with dzc
    k = K.build({'O1-F05': _tb('O1-F05')}, nguong=NG, a6=_a6(), ses=SES, cam=CAMF, a_m=0.1445,
                nhanh_tick='bo_tick', muc_kha_thi=['O1-F05'])
    assert k['eps_H'] == dict(mm=2.0, huong_deg=0.5)
    mc = k['mien_car_pose']
    assert mc['trang_thai'] == 'dat' and mc['side_m'] == 0.1445 and mc['phien_ban'] == 'car_pose_v2'
    assert mc['h_m'] == pytest.approx([0.1232 - 0.0012, 0.1250 + 0.0012])
    assert mc['kappa'] == pytest.approx([(0.1232 - 0.0012) / (2.632 + 0.01), (0.1250 + 0.0012) / (2.632 - 0.01)])
    # missing parts: null with the reason, not lockable
    k2 = K.build({'O1-F05': _tb('O1-F05')}, a6=_a6(dat=False), ses=dict(SES, car_pose_version=None), cam={})
    m2 = k2['mien_car_pose']
    assert m2['trang_thai'] == 'thieu' and m2['h_m'] is None and m2['kappa'] is None and m2['phien_ban'] is None
    assert any('car_pose_version' in w for w in m2['ly_do']) and any('dzc' in w for w in m2['ly_do'])
    assert k2['eps_H'] is None and 'eps_H' in k2['ly_do_chua_khoa'] and 'mien_car_pose' in k2['ly_do_chua_khoa']
    with pytest.raises(K.KhoaError):
        K.build({'O1-F05': _tb('O1-F05')}, nguong=dict(NG, format='x'))
    # CLI: the four files in khoa_truoc.nguon with their roles
    p = tmp_path / '3b.json'
    p.write_text(json.dumps(_tb('O1-F05')), encoding='utf-8')
    files = {}
    for nm, d in (('nguong', NG), ('a6', _a6()), ('session', SES), ('camera', CAMF)):
        files[nm] = tmp_path / f'{nm}.json'
        files[nm].write_text(json.dumps(d), encoding='utf-8')
    out = tmp_path / 'k.json'
    assert K.main(['--tb', f'O1-F05={p}', '--muc', 'O1-F05', '--canh-ma-m', '0.1445', '--nhanh-tick', 'bo_tick']
                  + sum([[f'--{nm}', str(fp)] for nm, fp in files.items()], []) + ['--out', str(out)]) == 0
    d = json.loads(out.read_text(encoding='utf-8'))
    assert {n['vai'] for n in d['khoa_truoc']['nguon']} >= {'nguong', 'a6', 'session', 'camera'}
    assert d['mien_car_pose']['trang_thai'] == 'dat'


def test_da_session_mien_car_pose(tmp_path):
    # da session (A), contract log 20/11 khuya: three camera sessions in session 2; they agree -> one domain,
    # dzc the largest camera file; a session that differs -> thieu with the field named
    one = K.mien_car_pose(_a6(), SES, CAMF)
    three = K.mien_car_pose(_a6(), [SES, dict(SES), dict(SES)],
                            [CAMF, dict(day_doi=dict(dzc_m=0.02)), dict(day_doi=dict(dzc_m=0.005))])
    assert three['trang_thai'] == 'dat' and three['side_m'] == one['side_m'] and three['h_m'] == one['h_m']
    assert three['kappa'] == pytest.approx([(0.1232 - 0.0012) / (2.632 + 0.02), (0.1250 + 0.0012) / (2.632 - 0.02)])
    assert K.mien_car_pose(_a6(), [SES], [CAMF]) == one
    for key, val in (('car_side_m', 0.145), ('car_pose_version', 'car_pose_v3'), ('kappa', 0.05)):
        m = K.mien_car_pose(_a6(), [SES, dict(SES, **{key: val})], [CAMF])
        assert m['trang_thai'] == 'thieu' and any(key in w for w in m['ly_do']), (key, m['ly_do'])
    m = K.mien_car_pose(_a6(), [SES], [CAMF, dict(day_doi={})])
    assert m['trang_thai'] == 'thieu' and m['kappa'] is None and any('dzc' in w for w in m['ly_do'])
    # CLI: --session, --camera repeated, every file in khoa_truoc.nguon
    p = tmp_path / '3b.json'
    p.write_text(json.dumps(_tb('O1-F05')), encoding='utf-8')
    args = []
    for i, (nm, d) in enumerate((('session', SES), ('session', SES), ('camera', CAMF),
                                 ('camera', dict(day_doi=dict(dzc_m=0.02))), ('a6', _a6()), ('nguong', NG))):
        f = tmp_path / f'{nm}{i}.json'
        f.write_text(json.dumps(d), encoding='utf-8')
        args += [f'--{nm}', str(f)]
    out = tmp_path / 'k.json'
    assert K.main(['--tb', f'O1-F05={p}', '--muc', 'O1-F05', '--canh-ma-m', '0.1445', '--nhanh-tick', 'bo_tick']
                  + args + ['--out', str(out)]) == 0
    d = json.loads(out.read_text(encoding='utf-8'))
    assert [n['vai'] for n in d['khoa_truoc']['nguon']].count('session') == 2
    assert [n['vai'] for n in d['khoa_truoc']['nguon']].count('camera') == 2
    assert d['mien_car_pose']['kappa'][1] == pytest.approx((0.1250 + 0.0012) / (2.632 - 0.02))


def test_commit_hil_with_bo_tick():
    """CODE 23/11 (Chot_T12_khoa_C8B_v1 muc 3; O1 v11.34 Q4): bo_tick needs the 40-hex scoring commit, written at the
    top and in khoa_truoc; giu_tick does not; a bad value is refused at the CLI."""
    ng = [dict(vai='ke_hoach_c8', file='k', sha256='0' * 64)]
    k = K.build({}, nguon=ng, nhanh_tick='bo_tick')
    assert k['trang_thai'] == 'chua_khoa' and 'commit_hil' in k['ly_do_chua_khoa']
    c = 'a' * 40
    k = K.build({}, nguon=ng, nhanh_tick='bo_tick', commit_hil=c)
    assert k['commit_hil'] == c and k['khoa_truoc']['commit_hil'] == c and 'commit_hil' not in k['ly_do_chua_khoa']
    assert 'commit_hil' not in K.build({}, nguon=ng, nhanh_tick='giu_tick')['ly_do_chua_khoa']
    assert not K._hex40('A' * 40) and not K._hex40('a' * 39) and K._hex40('0123456789abcdef' * 2 + '01234567')


def test_cli_refuses_bad_commit_hil(tmp_path, capsys):
    tb = tmp_path / 'tb.json'
    tb.write_text(json.dumps(_tb('O1-F05')), encoding='utf-8')
    rc = K.main(['--tb', f'O1-F05={tb}', '--commit-hil', 'xyz', '--out', str(tmp_path / 'o.json')])
    assert rc == 2 and not (tmp_path / 'o.json').exists()


def _kc(a3=0.2, p1=0.004, p7=1e-4, a5=0.13, th=2e-5, flag=False):
    return dict(A3_tuong_doi=dict(A3_xp_deg=dict(goc=a3), ngoai_suy=dict(goc=flag), mot_phuong=dict(goc=False),
                                  cap_hong=dict(goc=False)),
                P1_tuong_doi=dict(theo_vung=dict(goc=dict(max_m=p1, ngoai_suy=False, mot_phuong=False,
                                                          cap_hong=False))),
                P7=dict(lech_max_m=p7), A5=dict(lech_max_deg=a5), thang_chung=dict(lech_max=th))


def _src(**kw):
    a6 = dict(h_goc_m=[0.123, 0.122, 0.125, 0.126], u_h_goc_m=[0.013] * 4, H_m=2.632, h_m=0.124175,
              mien=dict(rho_max_m=1.615))
    d = dict(kiem_cheo=[_kc(), _kc(a3=0.25, p1=0.003, th=3e-6)], a6=a6,
             xoay=dict(cho=dict(goc=dict(trang_thai='dat', q_tag_m=[0.0003, 0.0004], u_q_m=0.0013))),
             u_h=dict(trang_thai='dat', u_h_chung_m=0.008), cam=[dict(day_doi=dict(dzc_m=0.017))])
    d.update(kw)
    return d


def test_hang_dong_from_the_sources(monkeypatch):
    """CODE 23/11 (Chot_T12_khoa_C8B_v1 muc 5): he_so_r from U_thang, P6 common, P5; hang_rad and hang_m stay null
    while a source is missing (never a partial sum); the parts read are the largest over the segments."""
    import c8b_a6 as a6m
    monkeypatch.setattr(a6m, 'a6du_rad', lambda a6, rho: 0.0078 if abs(rho - (1.615 - 0.1445 / math.sqrt(2))) < 1e-12
                        else None)
    v, why, part, _ = K.hang_dong_tinh(_src(), 0.1445)
    H, hmax = 2.632, 0.126 + 0.013
    assert part['A3_rad'] == pytest.approx(math.radians(0.25)) and part['P1_m'] == 0.004
    assert part['U_thang'] == pytest.approx(K.U_THANG_THUOC + 2e-5)
    want = K.U_THANG_THUOC + 2e-5 + 0.008 / (H - hmax) + hmax * 0.017 / ((H - hmax) * (H - 0.017))
    assert v['he_so_r'] == pytest.approx(want, rel=1e-12) and 'he_so_r' not in why
    assert part['thi_sai_m'] == pytest.approx(4 * 0.1445 / math.sqrt(2) * 0.005 / (4 * (H - 0.124175)))
    assert part['dq_khong_u_theta_m'] == pytest.approx(0.0013 + K.DQ_LECH_TAM_M) and part['q_hat_m'] == pytest.approx(5e-4)
    assert part['A6du_rad'] == 0.0078
    assert v['hang_rad'] is None and any('neo' in w for w in why['hang_rad'])
    assert v['hang_m'] is None and any('u_theta' in w for w in why['hang_m']) and v['u_phi_rad'] is None
    k = K.build({'O1-F05': _tb('O1-F05')}, a_m=0.1445, **{kk: vv for kk, vv in _src().items() if kk != 'a6'},
                a6=None)
    assert k['muc']['O1-F05']['hang_dong']['he_so_r'] is None                   # no A6: no P6 / P5 term


def test_hang_dong_refuses_flags_and_missing(monkeypatch):
    import c8b_a6 as a6m
    monkeypatch.setattr(a6m, 'a6du_rad', lambda a6, rho: 0.0078)
    v, why, _, _ = K.hang_dong_tinh(_src(kiem_cheo=[_kc(flag=True)]), 0.1445)
    assert any('ngoai_suy' in w for w in why['hang_rad'])
    v, why, _, _ = K.hang_dong_tinh(_src(cam=[]), 0.1445)
    assert v['he_so_r'] is None and any('dzc' in w for w in why['he_so_r'])
    v, why, _, _ = K.hang_dong_tinh(_src(u_h=dict(trang_thai='thieu', u_h_chung_m=0.008)), 0.1445)
    assert v['he_so_r'] is None
    v, why, _, _ = K.hang_dong_tinh(_src(kiem_cheo=[]), 0.1445)
    assert v['he_so_r'] is None and any('kiem_cheo' in w for w in why['he_so_r'])


def test_build_puts_the_margin_once(monkeypatch):
    import c8b_a6 as a6m
    monkeypatch.setattr(a6m, 'a6du_rad', lambda a6, rho: 0.0078)
    s = _src()
    k = K.build({'O1-F05': _tb('O1-F05')}, a_m=0.1445, kiem_cheo=s['kiem_cheo'], a6=s['a6'], xoay=s['xoay'],
                u_h=s['u_h'], cam=s['cam'])
    m = k['muc']['O1-F05']
    assert m['hang_dong']['he_so_r'] == pytest.approx(K.LE * m['hang_dong_truoc_le']['he_so_r'])
    assert m['hang_dong']['hang_m'] is None and 'u_theta' in m['hang_dong_ly_do']['hang_m']


def _f8null():
    return _f8([dict(luot_id='r1', cho='goc', lan_lap_id=None), dict(luot_id='r2', cho='K1', lan_lap_id=None)])


def _kc3(a3=dict(goc=0.2, K1=0.08, K2=0.19), flag=None):
    k = _kc()
    k['A3_tuong_doi'] = dict(A3_xp_deg=dict(a3), ngoai_suy={c: c == flag for c in K.CHO},
                             mot_phuong={c: False for c in K.CHO}, cap_hong={c: False for c in K.CHO})
    return k


def _tc(tan):
    return dict(schema='tham_chieu_v1', psi_tan_rad=tan)


def test_cach_a_u_beta_cam_and_p6_i(monkeypatch):
    """CODE 23/11 chot T12 muc 4, cach A (Liam at CODE 'Cach A (Recommended)'), correction 77991725: F8 all null ->
    u_beta_cam[k] = LE (2 A3_xp[k] + 2 A6du + largest psi_tan of the W at k), NOT 0; a missing part -> null at k,
    thieu; P6 rieng (i) 0; an F8 with ids keeps the beta_cam_replay rule."""
    import c8b_a6 as a6m
    monkeypatch.setattr(a6m, 'a6du_rad', lambda a6, rho: 0.001)
    s = _src()
    kcs = [_kc3(), _kc3(a3=dict(goc=0.25, K1=0.07, K2=0.1))]
    tcs = dict(goc=[_tc(0.002), _tc(0.003)], K1=[_tc(0.001)], K2=[_tc(0.0025)])
    ub, st, why, gd, stc = K.u_beta_cam(_f8null(), kcs, s['a6'], 0.1445, tcs)
    assert st == 'dat' and not why and gd['nhan'] == K.LL_E_NULL and stc == dict(goc='dat', K1='dat', K2='dat')
    assert ub['goc'] == pytest.approx(1.25 * math.degrees(2 * math.radians(0.25) + 2 * 0.001 + 0.003), rel=1e-12)
    assert ub['K1'] == pytest.approx(1.25 * math.degrees(2 * math.radians(0.08) + 2 * 0.001 + 0.001), rel=1e-12)
    assert all(v > 0 for v in ub.values())                                   # not 0 any more
    ub, st, why, _, stc = K.u_beta_cam(_f8null(), kcs, s['a6'], 0.1445, dict(goc=tcs['goc']))
    assert st == 'thieu' and ub['K1'] is None and stc['goc'] == 'dat' and any('K1' in w for w in why)
    ub, _, why, _, stc = K.u_beta_cam(_f8null(), [_kc3(flag='K1')], s['a6'], 0.1445, tcs)
    assert ub['K1'] is None and ub['goc'] is not None and any('ngoai_suy' in w for w in why)
    ub, _, _, _, _ = K.u_beta_cam(_f8null(), kcs, s['a6'], 0.1445, dict(tcs, goc=[_tc(None)]))
    assert ub['goc'] is None
    ub, _, _, _, _ = K.u_beta_cam(_f8null(), kcs, None, 0.1445, tcs)
    assert all(v is None for v in ub.values())
    ub, st, _, gd, _ = K.u_beta_cam(_f8())
    assert st == 'thieu' and gd is None
    _, why, part, _ = K.hang_dong_tinh(_src(lan_lap=_f8null()), 0.1445)
    assert part['P6_rieng_i_m'] == 0.0 and not any('P6 rieng dau (i)' in w for w in why['hang_m'])
    _, why, part, _ = K.hang_dong_tinh(_src(lan_lap=_f8()), 0.1445)
    assert 'P6_rieng_i_m' not in part and any('P6 rieng dau (i)' in w for w in why['hang_m'])


def _f7(st='khoa', uphi=0.5, uak=0.4):
    return dict(schema='truc_xe_v1', cho='goc', trang_thai=st, u_a_k_deg=uak, u_phi_hat_chuan_k_deg=uphi)


def test_full_hang_m_hang_rad_u_phi_and_lever(monkeypatch):
    import c8b_a6 as a6m
    monkeypatch.setattr(a6m, 'a6du_rad', lambda a6, rho: 0.0078)
    full = dict(lan_lap=_f8null(), truc_xe=_f7(),
                neo=dict(schema='c7_neo_luot_v1', theo_cho=dict(goc=dict(trang_thai='dat', P_neo_m=3e-4,
                                                                          A_neo_rad=2e-4))),
                u_moc=[dict(U_moc_hieu_xy_m=1e-4, U_moc_hieu_psi_rad=1e-4),
                       dict(U_moc_hieu_xy_m=2e-4, U_moc_hieu_psi_rad=5e-5)],
                W=dict(W_xy_m=5e-4, W_psi_rad=3e-4), P6_rieng_ii=dict(gia_tri_m=6e-4))
    v, why, p, lever = K.hang_dong_tinh(_src(**full), 0.1445)
    assert not why, why
    dq0 = 0.0013 + K.DQ_LECH_TAM_M
    dq = dq0 + (5e-4 + dq0) * math.radians(0.4)
    assert p['dq_m'] == pytest.approx(dq) and lever == pytest.approx(5e-4 + dq)
    a3, a5 = math.radians(0.25), math.radians(0.13)
    want_m = (0.004 + lever * (2 * a3 + 2 * 0.0078 + a5) + 2 * p['thi_sai_m'] + 1e-4 + 2 * dq + 0.0 + 6e-4 + 3e-4
              + 2e-4 + 5e-4)
    assert v['hang_m'] == pytest.approx(want_m, rel=1e-12)
    assert v['hang_rad'] == pytest.approx(2 * a3 + 2 * 0.0078 + a5 + 2e-4 + 1e-4 + 3e-4, rel=1e-12)
    assert v['u_phi_rad'] == pytest.approx(math.radians(0.5 + K.U_BETA_LECH_TAM_DEG))
    for drop in ('neo', 'u_moc', 'W', 'P6_rieng_ii', 'truc_xe'):
        v2, _, _, _ = K.hang_dong_tinh(_src(**dict(full, **{drop: None})), 0.1445)
        assert v2['hang_m'] is None, drop
    v3, _, _, _ = K.hang_dong_tinh(_src(**dict(full, truc_xe=_f7(st='chan_doan'))), 0.1445)
    assert v3['u_phi_rad'] is None and v3['hang_m'] is None
    v4, _, _, _ = K.hang_dong_tinh(_src(**dict(full, lan_lap=_f8())), 0.1445)
    assert v4['u_phi_rad'] is None                                       # cach A only when F8 all null
    s = _src(**full)
    kw = dict(lan_lap=s['lan_lap'], a_m=0.1445, kiem_cheo=s['kiem_cheo'], a6=s['a6'], xoay=s['xoay'], u_h=s['u_h'],
              cam=s['cam'], truc_xe=s['truc_xe'], neo=s['neo'], u_moc=s['u_moc'])
    k0 = K.build({'O1-F05': _tb('O1-F05')}, **kw)                        # no W of goc: u_beta_cam goc thieu
    assert k0['muc']['O1-F05']['hang_dong']['u_phi_rad'] is None
    assert 'u_beta_cam goc' in k0['muc']['O1-F05']['hang_dong_ly_do']['u_phi_rad']
    tcs = dict(goc=[_tc(0.002)], K1=[_tc(0.001)], K2=[_tc(0.001)])
    k = K.build({'O1-F05': _tb('O1-F05')}, tham_chieu=tcs, **kw)
    m = k['muc']['O1-F05']
    ubg = k['u_beta_cam_deg']['goc']
    assert ubg == pytest.approx(1.25 * math.degrees(2 * a3 + 2 * 0.0078 + 0.002), rel=1e-12)
    assert m['hang_dong']['u_phi_rad'] == pytest.approx(K.LE * math.radians(0.5 + K.U_BETA_LECH_TAM_DEG)
                                                        + math.radians(ubg), rel=1e-12)
    b = m['hang_theo_u'][0]
    assert b['P4_tay_don_m'] == pytest.approx(m['P4_tay_don']['L_m'] * math.radians(b['A4_deg']))
    assert b['P4_m'] == pytest.approx(m['can_quay']['p_m'] + b['P4_tay_don_m'], rel=1e-12)
    assert k['u_beta_cam_trang_thai_cho']['goc'] == 'dat' and k['u_beta_cam_gia_dinh']['nhan'] == K.LL_E_NULL
    assert k['u_beta_cam_trang_thai'] == 'thieu'           # _kc carries the A3 flags of goc only: K1, K2 thieu


def test_cli_tham_chieu_one_a6_and_place(tmp_path, capsys):
    """--tham-chieu CHO=path: a tham_chieu_v1 of another A6 than --a6 is refused (one A6 for the lock); a bad place
    or schema is refused; a good one is anchored in khoa_truoc.nguon."""
    tb = tmp_path / 'tb.json'
    tb.write_text(json.dumps(_tb('O1-F05')), encoding='utf-8')
    a6 = tmp_path / 'a6.json'
    a6.write_text(json.dumps(_src()['a6']), encoding='utf-8')
    import c8_u_imu as cu
    good = tmp_path / 'tc.json'
    good.write_text(json.dumps(dict(_tc(0.002), a6_sha256=cu.sha256(str(a6)))), encoding='utf-8')
    bad = tmp_path / 'tc_bad.json'
    bad.write_text(json.dumps(dict(_tc(0.002), a6_sha256='ab' * 32)), encoding='utf-8')
    base = ['--tb', f'O1-F05={tb}', '--a6', str(a6)]
    for i, extra in enumerate([['--tham-chieu', f'goc={bad}'], ['--tham-chieu', f'K3={good}'],
                               ['--tham-chieu', f'goc={good}', '--a6', str(a6)][:2] + ['--tham-chieu', f'K1={tb}']]):
        assert K.main(base + extra + ['--out', str(tmp_path / f'x{i}.json')]) == 2
        assert not (tmp_path / f'x{i}.json').exists()
    assert K.main(base + ['--tham-chieu', f'goc={good}', '--out', str(tmp_path / 'o.json')]) == 0
    k = json.loads((tmp_path / 'o.json').read_text(encoding='utf-8'))
    assert any(n['vai'] == 'tham_chieu goc' and n['sha256'] == cu.sha256(str(good)) for n in k['khoa_truoc']['nguon'])


def test_u_beta_cam_flag_missing_is_thieu(monkeypatch):
    """An A3 flag of the place absent (not False) counts as not checked: thieu at that place (design v7 section 1)."""
    import c8b_a6 as a6m
    monkeypatch.setattr(a6m, 'a6du_rad', lambda a6, rho: 0.001)
    k = _kc3()
    del k['A3_tuong_doi']['cap_hong']['K2']
    ub, _, why, _, stc = K.u_beta_cam(_f8null(), [k], _src()['a6'], 0.1445,
                                      dict(goc=[_tc(0.001)], K1=[_tc(0.001)], K2=[_tc(0.001)]))
    assert ub['K2'] is None and stc['K2'] == 'thieu' and ub['K1'] is not None and any('cap_hong' in w for w in why)


def _frames(path, rows):
    cols = 'idx,t_s,valid,tag_x,tag_y'
    path.write_text(cols + '\n' + '\n'.join(f'{i},{i / 60},{v},{x},{y}' for i, v, x, y in rows) + '\n',
                    encoding='utf-8')


def test_u_beta_cam_footprints(monkeypatch):
    """Cach A, 'hieu A6 du giua hai dau chan' (CODE 23/11, 77991725): A6du at the ref_12b footprint + A6du at the
    largest W footprint of the place, not 2 A6du at the edge of the domain; without footprints the domain bound."""
    import c8b_a6 as a6m
    monkeypatch.setattr(a6m, 'a6du_rad', lambda a6, rho: None if rho is None or rho > 1.55 else 0.001 * rho)
    kcs = [_kc3()]
    tcs = dict(goc=[_tc(0.002)], K1=[_tc(0.001)], K2=[_tc(0.001)])
    dc = dict(goc=dict(rho_12b_m=0.5, rho_W_m=0.6), K1=dict(rho_12b_m=1.6, rho_W_m=1.4))
    ub, _, why, gd, stc = K.u_beta_cam(_f8null(), kcs, _src()['a6'], 0.1445, tcs, dc)
    assert ub['goc'] == pytest.approx(1.25 * math.degrees(2 * math.radians(0.2) + 0.0011 + 0.002), rel=1e-12)
    assert ub['K1'] is None and stc['K1'] == 'thieu'                       # a footprint outside the domain
    rmax = 1.615 - 0.1445 / math.sqrt(2)
    assert ub['K2'] == pytest.approx(1.25 * math.degrees(2 * math.radians(0.19) + 2 * 0.001 * rmax + 0.001), rel=1e-12)
    assert gd['a6du_cach'] == dict(goc='dau_chan', K1='dau_chan', K2='rho_max (khong co dau chan; can tren)')


def test_rho_khung(tmp_path):
    import c8_u_imu as cu
    f = tmp_path / 'frames.csv'
    _frames(f, [(1, 1, 0.3, 0.4), (2, 1, 0.6, 0.8), (3, 0, 9.0, 9.0), (4, 1, 0.0, 0.1)])
    sha = cu.sha256(str(f))
    assert K.rho_khung(str(f), [1, 2], sha, [0.0, 0.0]) == pytest.approx(1.0)
    assert K.rho_khung(str(f), [1, 4], sha, [0.0, 0.1]) == pytest.approx(math.hypot(0.3, 0.3))
    for ids, s in (([1, 3], sha), ([1, 5], sha), ([1], 'ab' * 32), ([], sha)):
        with pytest.raises(K.KhoaError):
            K.rho_khung(str(f), ids, s, [0.0, 0.0])


def test_cli_dau_chan(tmp_path, monkeypatch):
    """--cua-so F2 + --khung-12b CHO=frames.csv: the F2 ref_12b window of that place on those frames and the W ids of
    the --tham-chieu frames give u_beta_cam_dau_chan; a frames file with no F2 window is refused."""
    import c8_u_imu as cu
    tb = tmp_path / 'tb.json'
    tb.write_text(json.dumps(_tb('O1-F05')), encoding='utf-8')
    a6 = tmp_path / 'a6.json'
    a6.write_text(json.dumps(dict(_src()['a6'], plumb_xy_m=[0.0, 0.0])), encoding='utf-8')
    f12, fW = tmp_path / 'f12.csv', tmp_path / 'fW.csv'
    _frames(f12, [(10, 1, 0.3, 0.4), (11, 1, 0.3, 0.41), (12, 1, 5.0, 5.0)])
    _frames(fW, [(7, 1, 0.6, 0.0), (8, 1, 0.61, 0.0)])
    f2 = tmp_path / 'cua_so.jsonl'
    f2.write_text('\n'.join(json.dumps(r) for r in (
        dict(loai='ref_12b', cho='goc', luot_id='HC_goc_2', frames_sha256=cu.sha256(str(f12)),
             khung=[dict(id=10), dict(id=11)]),
        dict(loai='ref_12b', cho='K1', luot_id='HC_K1_2', frames_sha256=cu.sha256(str(f12)),
             khung=[dict(id=12)]))) + '\n', encoding='utf-8')
    tc = tmp_path / 'tc.json'
    tc.write_text(json.dumps(dict(_tc(0.002), a6_sha256=cu.sha256(str(a6)), ids=[7, 8],
                                  frames_sha256=cu.sha256(str(fW)),
                                  nguon=[dict(vai_tro='frames', duong_dan=str(fW), sha256=cu.sha256(str(fW)))])),
                  encoding='utf-8')
    base = ['--tb', f'O1-F05={tb}', '--a6', str(a6), '--tham-chieu', f'goc={tc}', '--cua-so', str(f2)]
    assert K.main(base + ['--khung-12b', f'goc={f12}', '--out', str(tmp_path / 'o.json')]) == 0
    k = json.loads((tmp_path / 'o.json').read_text(encoding='utf-8'))
    d = k['u_beta_cam_dau_chan']['goc']
    assert d['rho_12b_m'] == pytest.approx(math.hypot(0.3, 0.41)) and d['rho_W_m'] == pytest.approx(0.61)
    assert d['luot_12b'] == ['HC_goc_2'] and any(n['vai'] == 'khung_12b goc' for n in k['khoa_truoc']['nguon'])
    assert K.main(base + ['--khung-12b', f'K2={f12}', '--out', str(tmp_path / 'x.json')]) == 2   # no K2 window
    assert K.main(['--tb', f'O1-F05={tb}', '--khung-12b', f'goc={f12}', '--out', str(tmp_path / 'y.json')]) == 2
