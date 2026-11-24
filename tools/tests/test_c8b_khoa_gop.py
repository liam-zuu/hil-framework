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
                       hang_theo_u=[dict(u_lo_m_s=0.0, u_hi_m_s=0.358, dai='B1', A4_deg=0.6,
                                         P4_m=0.004 + K.LE * 0.003 * s + 0.04 * math.radians(0.6),   # Codex KG7-1
                                         P6_rieng_ii_m=0.003 * s, trang_thai='gioi_han', khong_khung=False,
                                         P4_tay_don_m=0.04 * math.radians(0.6))],
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
                                       dict(vai='nguong', file='ng.json', sha256='4' * 64),
                                       dict(vai='3b O1-F02', file='3b.json', sha256='5' * 64),
                                       dict(vai='truc_xe', file='tx.json', sha256='6' * 64),
                                       dict(vai='session', file='session.json', sha256=sess),
                                       dict(vai='kiem_cheo', file='kc.json', sha256=sess[:-1] + 'a'),
                                       dict(vai='tham_chieu goc', file='tc.json', sha256=sess[:-1] + 'b')],
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
    assert m['hang_theo_u'][0]['P4_m'] == pytest.approx(0.004 + K.LE * 0.003 * 1.02 + 0.04 * math.radians(0.6))
    assert m['hang_theo_u'][0]['trang_thai'] == 'gioi_han'
    assert m['hang_dong_thanh_phan'] == dict(theo_buoi={'2026-11-20_run02': dict(W_n_luot=10),
                                                        '2026-11-23_run03': dict(W_n_luot=2)})
    assert k['u_beta_cam_deg'] == dict(goc=1.174, K1=None, K2=None)
    assert k['u_beta_cam_trang_thai'] == dict(goc='dat', K1='thieu', K2='thieu')     # what c7_score reads
    assert k['mien_car_pose']['kappa'] == [0.0413, 0.0531]
    mb = k['mien_car_pose_theo_buoi']                       # KG-5: per session, keyed by the gate's session id
    assert sorted(mb) == ['C8_2026-11-20_run02', 'C8_2026-11-23_run03']
    assert mb['C8_2026-11-23_run03']['kappa'] == [0.0414, 0.0530] and set(mb['C8_2026-11-20_run02']) == {
        'side_m', 'phien_ban', 'h_m', 'kappa'}
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


@pytest.mark.parametrize('bad', ['bang', 'eps', 'a6', 'plan', 'band', 'top', 'nguong', '3b', 'truc_xe', 'a6_xoa',
                                 'them_vai'])
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
    elif bad in ('nguong', '3b', 'truc_xe'):       # Codex KG-2, KG-3: same numbers, another common source
        i = {'nguong': 2, '3b': 3, 'truc_xe': 4}[bad]
        k['khoa_truoc']['nguon'][i]['sha256'] = 'e' * 64
    elif bad == 'a6_xoa':                          # KG-2: one lock without its A6 entry (the other has it)
        del k['khoa_truoc']['nguon'][0]
    elif bad == 'them_vai':
        k['khoa_truoc']['nguon'].append(dict(vai='xoay', file='x.json', sha256='7' * 64))
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


@pytest.mark.parametrize('f', ['band', 'u_beta', 'mien'])
def test_states_only_accepted(f):
    """Codex KG-8: a state the gate / scorer does not accept for that field is refused, even if the same in both."""
    kb = _two()
    for _, k in kb:
        if f == 'band':
            k['muc']['O1-F05']['hang_theo_u'][0]['trang_thai'] = 'chua_ket_luan'
        elif f == 'u_beta':
            k['u_beta_cam_trang_thai']['goc'] = 'gioi_han'
        else:
            k['mien_car_pose']['trang_thai'] = 'chua_ket_luan'
    with pytest.raises(G.GopError, match='accepted'):
        G.gop(kb, CH)


def test_per_session_sources_may_differ():
    kb = _two()
    vai = [n['vai'] for n in G.gop(kb, CH)['khoa_truoc']['nguon']]
    assert vai.count('session') == 2 and vai.count('kiem_cheo') == 2 and vai.count('tham_chieu goc') == 2
    assert vai.count('truc_xe') == 1 and vai.count('nguong') == 1


# ── a session without frames in a speed band (CODE 24/11 khuya, 2026-11-24_CODE_chot_khoa_gop_dai_khong_co_khung) ──

EDGES = (0.0, 0.358, 0.896, 1.433)


def _ba_dai(k, s, khong=(), lv='O1-F08'):
    """Three bands B1..B3 at a level of a session lock (terms scaled by s); the bands of `khong`: no frame of the
    session there (as c8b_khoa_v5.hang_theo_u writes them: thieu, null, no lever key)."""
    bands = []
    for i, nm in enumerate(K.P6_BANDS):
        b = dict(u_lo_m_s=EDGES[i], u_hi_m_s=EDGES[i + 1], dai=nm)
        if nm in khong:
            b.update(P4_m=None, A4_deg=None, P6_rieng_ii_m=None, trang_thai='thieu', khong_khung=True)
        else:
            b.update(P4_m=0.004 + K.LE * 0.003 * s * (i + 1) + 0.04 * math.radians(0.6), A4_deg=0.6,
                     P6_rieng_ii_m=0.003 * s * (i + 1), trang_thai='gioi_han', khong_khung=False,
                     P4_tay_don_m=0.04 * math.radians(0.6))
        bands.append(b)
    k['muc'][lv]['hang_theo_u'] = bands
    return k


def _ba_buoi():
    """Session 2 (every level), run05 (F05 runs only: no frame in B3), run06 (F02 + F08: every band)."""
    return [('2026-11-20_run02', _ba_dai(_khoa(1.0, 1.058, sess='1' * 64), 1.0)),
            ('2026-11-24_run05', _ba_dai(_khoa(1.5, 1.174, sess='2' * 64), 1.5, khong=('B3',))),
            ('2026-11-24_run06', _ba_dai(_khoa(1.2, 1.1, sess='3' * 64), 1.2))]


def test_session_without_frames_gives_nothing_to_the_band():
    k = G.gop(_ba_buoi(), CH)
    b = k['muc']['O1-F08']['hang_theo_u']
    p4 = lambda s, i: 0.004 + K.LE * 0.003 * s * (i + 1) + 0.04 * math.radians(0.6)   # noqa: E731
    assert b[0]['P4_m'] == pytest.approx(p4(1.5, 0)) and b[1]['P4_m'] == pytest.approx(p4(1.5, 1))   # run05 the largest
    assert b[2]['P4_m'] == pytest.approx(p4(1.2, 2)) and b[2]['P6_rieng_ii_m'] == pytest.approx(0.003 * 1.2 * 3)
    assert b[2]['A4_deg'] == 0.6 and b[2]['trang_thai'] == 'gioi_han' and b[2]['khong_khung'] is False
    assert all(x['khong_khung'] is False and x['P4_m'] is not None for x in b)
    assert [(x['u_lo_m_s'], x['u_hi_m_s'], x['dai']) for x in b] == [(EDGES[i], EDGES[i + 1], nm)
                                                                    for i, nm in enumerate(K.P6_BANDS)]
    assert k['gop']['lon_nhat_tu']['muc.O1-F08.hang_theo_u[2].P4_m'] == '2026-11-24_run06'
    assert k['gop']['lon_nhat_tu']['muc.O1-F08.hang_theo_u[1].P4_m'] == '2026-11-24_run05'
    d = k['gop']['dai_theo_buoi']['O1-F08']
    assert d[2] == dict(u_lo_m_s=EDGES[2], u_hi_m_s=EDGES[3], dai='B3',
                        dong_gop=['2026-11-20_run02', '2026-11-24_run06'], khong_khung=['2026-11-24_run05'])
    assert d[0]['khong_khung'] == [] and len(d[0]['dong_gop']) == 3
    assert k['gop']['dai_theo_buoi']['O1-F05'] == [dict(u_lo_m_s=0.0, u_hi_m_s=0.358, dai='B1', khong_khung=[],
                                                        dong_gop=[b_ for b_, _ in _ba_buoi()])]
    assert k['muc_du_dai_theo_buoi'] == {'C8_2026-11-20_run02': list(K.LEVELS),
                                         'C8_2026-11-24_run05': ['O1-F02', 'O1-F05'],
                                         'C8_2026-11-24_run06': list(K.LEVELS)}
    assert k['trang_thai'] == 'khoa' and 'khong dong gop' in k['gop']['quy_tac']
    # the order of the sessions changes nothing
    k2 = G.gop(_ba_buoi()[::-1], CH)
    assert k2['muc']['O1-F08']['hang_theo_u'] == b and k2['muc_du_dai_theo_buoi'] == k['muc_du_dai_theo_buoi']
    # a level outside muc_kha_thi may keep a band nobody has frames in (null, as before it was not looked at)
    kb = _ba_buoi()
    for _, x in kb:
        x['muc_kha_thi'] = ['O1-F02', 'O1-F05']
        _ba_dai(x, 1.0, khong=('B3',))
    k3 = G.gop(kb, CH)
    assert k3['muc']['O1-F08']['hang_theo_u'][2] == dict(u_lo_m_s=EDGES[2], u_hi_m_s=EDGES[3], dai='B3', P4_m=None,
                                                         A4_deg=None, P6_rieng_ii_m=None, trang_thai='thieu',
                                                         khong_khung=True)
    assert all('O1-F08' not in v for v in k3['muc_du_dai_theo_buoi'].values())


@pytest.mark.parametrize('ca', ['khong_ai', 'co_khung_thieu', 'kk_co_so', 'kk_trang_thai', 'thieu_co', 'co_la', 'mep',
                                'mot_buoi'])
def test_band_cases_refused(ca):
    """(b) a band no session has a frame in: null, refused as before; item 3: a session WITH frames and no number
    still makes the band null (refused); a flag that contradicts the band, a lock without the flag, other edges."""
    kb = _ba_buoi()
    k5 = kb[1][1]
    b3 = k5['muc']['O1-F08']['hang_theo_u'][2]
    if ca == 'khong_ai':
        for _, x in kb:
            _ba_dai(x, 1.0, khong=('B3',))
    elif ca == 'co_khung_thieu':
        b3['khong_khung'] = False                        # frames without a number: not concluded in one session
    elif ca == 'kk_co_so':
        b3['P4_m'] = 0.01
    elif ca == 'kk_trang_thai':
        b3['trang_thai'] = 'gioi_han'
    elif ca == 'thieu_co':
        del kb[0][1]['muc']['O1-F08']['hang_theo_u'][0]['khong_khung']
    elif ca == 'co_la':
        b3['khong_khung'] = 'true'
    elif ca == 'mep':
        b3['u_hi_m_s'] = 1.5
    else:
        kb = [kb[1]]                                     # one session, a band without frames at a level to lock
    with pytest.raises(G.GopError):
        G.gop(kb, CH)


def test_band_refusals_say_why():
    kb = _ba_buoi()
    for _, x in kb:
        _ba_dai(x, 1.0, khong=('B3',))
    with pytest.raises(G.GopError, match='no session has a frame'):
        G.gop(kb, CH)
    kb = _ba_buoi()
    del kb[2][1]['muc']['O1-F08']['hang_theo_u'][1]['khong_khung']
    with pytest.raises(G.GopError, match='khong_khung'):
        G.gop(kb, CH)
