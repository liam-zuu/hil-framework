"""Tests of c8b_khoa_gop (one C8b lock over several sessions, the largest term; Liam 24/11 A2 at CODE)."""
import copy
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import c8b_khoa_gop as G   # noqa: E402
import c8b_khoa_v5 as K    # noqa: E402

CH = 'a' * 40
A6 = 'b' * 64
KH = 'c' * 64
UPH = 0.01            # hang_dong_truoc_le.u_phi_rad (u_phi_hat of the one F7), the same in every session


def _khoa(s, ub_goc, kap=(0.0413, 0.0531), sess='d' * 64, buoi='x'):
    """A minimal locked c8b_khoa_v5 of one session: terms scaled by s."""
    muc = {}
    for lv in K.LEVELS:
        tr = dict(a_m=0.1445, hang_m=0.019 * s, he_so_r=0.0046 * s, hang_rad=0.032 * s, u_phi_rad=UPH)
        hd = dict(a_m=0.1445, hang_m=K.LE * tr['hang_m'], he_so_r=K.LE * tr['he_so_r'],
                  hang_rad=K.LE * tr['hang_rad'], u_phi_rad=K.LE * UPH + math.radians(ub_goc))
        muc[lv] = dict(can_quay=dict(trang_thai='dat', p_m=0.004, psi_deg=0.6), man_trap=dict(nhanh='mot_khung'),
                       so_do=['g5'] if lv == 'O1-F08' else [],
                       hang_theo_u=[dict(u_lo_m_s=0.0, u_hi_m_s=0.358, dai='B1', P4_m=0.007 * s, A4_deg=0.65,
                                         P6_rieng_ii_m=0.003 * s, trang_thai='gioi_han', P4_tay_don_m=2.6e-5)],
                       hang_theo_u_ly_do=[], hang_dong=hd, hang_dong_truoc_le=tr,
                       hang_dong_thanh_phan=dict(W_n_luot=int(10 * s)), P4_tay_don=dict(L_m=0.04, ly_do=None),
                       hang_dong_ly_do=dict(a_m=None, hang_m=None, he_so_r=None, u_phi_rad=None, hang_rad=None),
                       khac={})
    ust = dict(goc='dat', K1='thieu', K2='thieu')
    return dict(version=K.VERSION, le=K.LE, muc=muc, bang_man_trap={lv: dict(nhanh='mot_khung') for lv in K.LEVELS},
                u_beta_cam_deg=dict(goc=ub_goc, K1=None, K2=None), u_beta_cam_trang_thai=dict(ust),
                u_beta_cam_trang_thai_chung='thieu', u_beta_cam_trang_thai_cho=dict(ust),
                u_beta_cam_dau_chan=dict(goc=dict(n_W=int(10 * s))), u_beta_cam_gia_dinh=dict(nhan='x'),
                u_beta_cam_ly_do=['K1: thieu', 'K2: thieu'], nhanh_tick='bo_tick', commit_hil=CH,
                muc_kha_thi=list(K.LEVELS), eps_H=dict(mm=2.0, huong_deg=0.15), eps_H_ly_do=None,
                mien_car_pose=dict(side_m=0.1445, phien_ban='car_pose_v2', h_m=[0.109, 0.139], kappa=list(kap),
                                   trang_thai='dat', ly_do=[]),
                lan_lap_E=dict(goc=dict(trang_thai='dat')), lan_lap_E_trang_thai='dat', lan_lap_E_ly_do=[],
                khoa_truoc=dict(nguon=[dict(vai='a6', file='a6.json', sha256=A6),
                                       dict(vai='ke_hoach_c8', file='kh.json', sha256=KH),
                                       dict(vai='session', file='session.json', sha256=sess)],
                                script=dict(commit=CH, dirty=False), commit_hil=CH, luc='2026-11-24T19:00:00+07:00'),
                trang_thai='khoa', ly_do_chua_khoa=None)


def _two():
    return [('2026-11-20_run02', _khoa(1.0, 1.058, sess='1' * 64)),
            ('2026-11-23_run03', _khoa(1.02, 1.174, kap=(0.0414, 0.0530), sess='2' * 64))]


def test_largest_terms_states_hull_sources():
    kb = _two()
    for lv in K.LEVELS:
        kb[1][1]['muc'][lv]['hang_dong_thanh_phan'] = dict(W_n_luot=2)
    k = G.gop(kb, CH, nguon_khoa={b: dict(file='k.json', duong_dan=f'/{b}/k.json', sha256=str(i) * 64)
                                  for i, (b, _) in enumerate(kb)})
    m = k['muc']['O1-F05']
    assert m['hang_dong']['hang_m'] == pytest.approx(K.LE * 0.019 * 1.02, rel=1e-12)
    assert m['hang_dong']['u_phi_rad'] == pytest.approx(K.LE * UPH + math.radians(1.174), rel=1e-12)
    assert m['hang_theo_u'][0]['P4_m'] == pytest.approx(0.007 * 1.02)
    assert m['hang_theo_u'][0]['trang_thai'] == 'gioi_han'
    assert m['hang_dong_thanh_phan'] == dict(theo_buoi={'2026-11-20_run02': dict(W_n_luot=10),
                                                        '2026-11-23_run03': dict(W_n_luot=2)})
    assert k['u_beta_cam_deg'] == dict(goc=1.174, K1=None, K2=None)
    assert k['u_beta_cam_trang_thai'] == dict(goc='dat', K1='thieu', K2='thieu')     # what c7_score reads
    assert k['mien_car_pose']['kappa'] == [0.0413, 0.0531]
    assert k['trang_thai'] == 'khoa' and k['version'] == K.VERSION and k['commit_hil'] == CH
    vai = [n['vai'] for n in k['khoa_truoc']['nguon']]
    assert vai.count('a6') == 1 and vai.count('ke_hoach_c8') == 1 and vai.count('session') == 2
    assert 'khoa_buoi 2026-11-20_run02' in vai and 'khoa_buoi 2026-11-23_run03' in vai
    assert k['gop']['lon_nhat_tu']['u_beta_cam_deg.goc'] == '2026-11-23_run03'
    assert k['gop']['lon_nhat_tu']['muc.O1-F02.hang_dong.hang_m'] == '2026-11-23_run03'
    assert k['u_beta_cam_ly_do'] == ['K1: thieu', 'K2: thieu']                         # same in both: kept
    # the mixed case: the largest from different sessions per term
    kb[0][1]['muc']['O1-F02']['hang_theo_u'][0]['P4_m'] = 0.0116
    k = G.gop(kb, CH)
    assert k['muc']['O1-F02']['hang_theo_u'][0]['P4_m'] == 0.0116
    assert k['gop']['lon_nhat_tu']['muc.O1-F02.hang_theo_u[0].P4_m'] == '2026-11-20_run02'


def test_one_session_is_itself():
    b, k0 = _two()[0]
    k = G.gop([(b, copy.deepcopy(k0))], CH)
    for lv in K.LEVELS:
        assert k['muc'][lv]['hang_dong'] == k0['muc'][lv]['hang_dong']
    assert k['u_beta_cam_deg'] == k0['u_beta_cam_deg'] and k['mien_car_pose'] == k0['mien_car_pose']


def test_per_session_reasons_prefixed():
    kb = _two()
    kb[1][1]['u_beta_cam_ly_do'] = ['K2: khac']
    k = G.gop(kb, CH)
    assert k['u_beta_cam_ly_do'] == ['2026-11-20_run02: K1: thieu', '2026-11-20_run02: K2: thieu',
                                     '2026-11-23_run03: K2: khac']


@pytest.mark.parametrize('bad', ['bang', 'eps', 'a6', 'plan', 'band', 'top'])
def test_must_agree(bad):
    kb = _two()
    k = kb[1][1]
    if bad == 'bang':
        k['bang_man_trap']['O1-F02'] = dict(nhanh='alpha')
    elif bad == 'eps':
        k['eps_H'] = dict(mm=2.5, huong_deg=0.15)
    elif bad == 'a6':
        k['khoa_truoc']['nguon'][0]['sha256'] = 'e' * 64
    elif bad == 'plan':
        k['khoa_truoc']['nguon'][1]['sha256'] = 'e' * 64
    elif bad == 'band':
        k['muc']['O1-F05']['hang_theo_u'][0]['u_hi_m_s'] = 0.4
    else:
        k['them'] = 1
    with pytest.raises(G.GopError):
        G.gop(kb, CH)


@pytest.mark.parametrize('bad', ['commit', 'commit_kt', 'chua_khoa', 'truoc_N1', 'version', 'ten'])
def test_inputs_refused(bad):
    kb = _two()
    k = kb[1][1]
    if bad == 'commit':
        k['commit_hil'] = 'f' * 40
    elif bad == 'commit_kt':
        k['khoa_truoc']['commit_hil'] = 'f' * 40
    elif bad == 'chua_khoa':
        k['trang_thai'] = 'chua_khoa'
    elif bad == 'truoc_N1':
        k['u_beta_cam_trang_thai'] = 'thieu'
        del k['u_beta_cam_trang_thai_chung']
    elif bad == 'version':
        k['version'] = 'c8b_khoa_v4'
    else:
        kb[1] = (kb[0][0], k)
    with pytest.raises(G.GopError):
        G.gop(kb, CH)


def test_u_phi_from_different_f7_refused():
    """u_phi_rad = LE x truoc_le.u_phi_rad + u_beta_cam goc must still hold: two u_phi_hat (two F7) with the largest
    u_beta_cam in the other session break it."""
    kb = _two()
    for lv in K.LEVELS:
        m = kb[0][1]['muc'][lv]
        m['hang_dong_truoc_le']['u_phi_rad'] = 0.02
        m['hang_dong']['u_phi_rad'] = K.LE * 0.02 + math.radians(1.058)
    with pytest.raises(G.GopError, match='u_phi'):
        G.gop(kb, CH)


def test_null_in_one_session():
    kb = _two()
    kb[0][1]['u_beta_cam_deg']['K1'] = 0.5
    kb[0][1]['u_beta_cam_trang_thai']['K1'] = kb[0][1]['u_beta_cam_trang_thai_cho']['K1'] = 'dat'
    k = G.gop(kb, CH)
    assert k['u_beta_cam_deg']['K1'] is None and k['u_beta_cam_trang_thai']['K1'] == 'thieu'
    kb[1][1]['muc']['O1-F05']['hang_dong']['hang_m'] = None
    kb[1][1]['muc']['O1-F05']['hang_dong_truoc_le']['hang_m'] = None
    with pytest.raises(G.GopError):          # the combined lock is not locked (hang_dong missing)
        G.gop(kb, CH)


def test_cli(tmp_path, capsys):
    ps = []
    for b, k in _two():
        p = tmp_path / f'{b}.json'
        p.write_text(json.dumps(k), encoding='utf-8')
        ps.append(f'{b}={p}')
    out = tmp_path / 'gop.json'
    argv = sum((['--khoa-buoi', s] for s in ps), []) + ['--commit-hil', CH, '--out', str(out)]
    assert G.main(argv) == 0 and capsys.readouterr().out.rstrip().endswith('#ket_qua,ma=0')
    k = json.loads(out.read_text(encoding='utf-8'))
    assert k['trang_thai'] == 'khoa' and len(k['gop']['khoa_buoi']) == 2
    assert all(len(v['sha256']) == 64 for v in k['gop']['khoa_buoi'].values())
    assert G.main(argv) == 2                                              # never overwrites
    assert G.main(argv[:-4] + ['--commit-hil', 'xyz', '--out', str(tmp_path / 'b.json')]) == 2
