"""c8b_p6_muon (L172, Liam 25/11 ~23:45 at CODE): the P6 rieng (ii) of a session with one run, borrowed as the largest
delta_h_m per band of the P6 files of other sessions; c8b_khoa_v5 takes it only for the session it was made for."""
import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import c8_u_imu as cu  # noqa: E402
import c8b_khoa_v5 as K  # noqa: E402
import c8b_p6_muon as M  # noqa: E402

EDGE = dict(B0=(0.0, 0.01), B1=(0.0, 0.35), B2=(0.35, 0.89), B3=(0.89, 1.43))
NG = 'a' * 64


def _p6(ses, vals, ng=NG, **kw):
    """A computed P6 file: vals {band: delta or None (thieu)}."""
    dai = {}
    for b in M.BANDS:
        v = vals.get(b)
        dai[b] = dict(delta_h_m=v, trang_thai='dat' if v is not None else 'thieu', n_khung=0 if v is None else 50,
                      n_do=0 if v is None else 9, u_lo_m_s=EDGE[b][0], u_hi_m_s=EDGE[b][1])
    d = dict(schema=M.P6_VERSION, dai=dai, trang_thai='dat' if all(v is not None for v in vals.values()) else 'thieu',
             luot=[dict(luot='C8-P-F05-1'), dict(luot='C8-P-F05-2')],
             nguon=[dict(vai='session', sha256=ses), dict(vai='nguong', sha256=ng)])
    d.update(kw)
    return d


A = _p6('1' * 64, dict(B0=0.004, B1=0.010, B2=0.011, B3=0.009))
B = _p6('2' * 64, dict(B0=0.005, B1=0.003, B2=0.015, B3=None))


def test_same_names_as_the_lock():
    assert M.P6_VERSION == K.P6_VERSION and M.BANDS == ('B0',) + K.P6_BANDS


def test_gop_largest_per_band_and_its_source():
    dai, tom, ng = M.gop([('c' * 64, A), ('d' * 64, B)])
    assert ng == NG
    assert {b: dai[b]['delta_h_m'] for b in M.BANDS} == dict(B0=0.005, B1=0.010, B2=0.015, B3=0.009)
    assert [dai[b]['muon_tu_sha256'][0] for b in M.BANDS] == ['d', 'c', 'd', 'c']
    assert [dai[b]['n_nguon'] for b in M.BANDS] == [2, 2, 2, 1] and all(dai[b]['trang_thai'] == 'dat' for b in M.BANDS)
    assert (dai['B2']['u_lo_m_s'], dai['B2']['u_hi_m_s']) == EDGE['B2']
    assert tom[1]['dai']['B3'] == dict(trang_thai='thieu', delta_h_m=None) and tom[1]['session_sha256'] == ['2' * 64]
    # the order of the sources does not change the numbers
    d2, _, _ = M.gop([('d' * 64, B), ('c' * 64, A)])
    assert d2 == dai
    # no source has the band: thieu, null (the lock stops there as with any P6 file)
    d3, _, _ = M.gop([('d' * 64, B)])
    assert d3['B3'] == dict(delta_h_m=None, trang_thai='thieu', n_nguon=0, muon_tu_sha256=None)


@pytest.mark.parametrize('ca', ['rong', 'trung_sha', 'trung_session', 'nguong_khac', 'da_muon', 'schema', 'khong_session',
                                'canh_khac', 'dat_khong_so', 'so_am', 'trang_thai_la', 'hai_nguong'])
def test_gop_refuses(ca):
    a, b = copy.deepcopy(A), copy.deepcopy(B)
    src = [('c' * 64, a), ('d' * 64, b)]
    if ca == 'rong':
        src = []
    elif ca == 'trung_sha':
        src = [('c' * 64, a), ('c' * 64, b)]
    elif ca == 'trung_session':
        b['nguon'][0]['sha256'] = '1' * 64
    elif ca == 'nguong_khac':
        b['nguon'][1]['sha256'] = 'b' * 64
    elif ca == 'da_muon':
        b['muon'] = dict(version=M.VERSION)
    elif ca == 'schema':
        b['schema'] = 'c8b_P6_rieng_ii_v1'
    elif ca == 'khong_session':
        b['nguon'] = b['nguon'][1:]
    elif ca == 'canh_khac':
        b['dai']['B1']['u_hi_m_s'] = 0.36
    elif ca == 'dat_khong_so':
        b['dai']['B1']['delta_h_m'] = None
    elif ca == 'so_am':
        b['dai']['B1']['delta_h_m'] = -1e-3
    elif ca == 'trang_thai_la':
        b['dai']['B3']['trang_thai'] = 'gioi_han'
    elif ca == 'hai_nguong':
        b['nguon'].append(dict(vai='nguong', sha256='b' * 64))
    with pytest.raises(M.MuonError):
        M.gop(src)


def _files(tmp_path, a=A, b=B):
    ps = []
    for nm, d in (('a', a), ('b', b)):
        os.makedirs(tmp_path / nm, exist_ok=True)
        p = tmp_path / nm / 'c8b_P6_rieng_ii_v2.json'
        p.write_text(json.dumps(d), encoding='utf-8')
        ps.append(str(p))
    ses = tmp_path / 'session.json'
    ses.write_text(json.dumps(dict(buoi='nhan')), encoding='utf-8')
    ng = tmp_path / 'nguong.json'
    ng.write_text(json.dumps(dict(format='nguong_khoa_c8_v1', eps_H=dict(mm=1.0, huong_deg=0.5))), encoding='utf-8')
    return ps, str(ses), str(ng)


def _made(tmp_path):
    """Sources whose nguong sha is the sha of the nguong file on disk; the borrowed file written by the CLI."""
    ng = tmp_path / 'nguong.json'
    ng.write_text(json.dumps(dict(format='nguong_khoa_c8_v1', eps_H=dict(mm=1.0, huong_deg=0.5))), encoding='utf-8')
    s = cu.sha256(str(ng))
    ps, ses, ng = _files(tmp_path, _p6('1' * 64, dict(B0=0.004, B1=0.010, B2=0.011, B3=0.009), ng=s),
                         _p6('2' * 64, dict(B0=0.005, B1=0.003, B2=0.015, B3=None), ng=s))
    out = tmp_path / 'muon.json'
    assert M.main(['--p6', ps[0], '--p6', ps[1], '--session-nhan', ses, '--nguong', ng,
                   '--quyet-dinh', 'L172 Liam 25/11', '--ly-do', 'mot luot', '--out', str(out)]) == 0
    return ps, ses, ng, str(out), json.loads(out.read_text(encoding='utf-8'))


def test_cli_writes_the_declared_file(tmp_path, capsys):
    ps, ses, ng, out, d = _made(tmp_path)
    assert capsys.readouterr().out.strip().endswith('#ket_qua,ma=0')
    assert d['schema'] == K.P6_VERSION and d['trang_thai'] == 'dat' and 'luot' not in d
    m = d['muon']
    assert m['version'] == M.VERSION and m['quy_tac'] == 'max_theo_dai' and m['quyet_dinh'] == 'L172 Liam 25/11'
    assert m['session_nhan_sha256'] == [cu.sha256(ses)] and [r['duong_dan'] for r in m['nguon_p6']] == \
        [os.path.abspath(p) for p in ps] and [r['sha256'] for r in m['nguon_p6']] == [cu.sha256(p) for p in ps]
    assert sorted(n['vai'] for n in d['nguon']) == ['nguong', 'p6_muon', 'p6_muon', 'session_nhan']
    assert d['script']['file'] == 'tools/c8b_p6_muon.py' and d['luc']
    # the lock reads it as any P6 file
    assert K.p6_delta(d, 'B0') == 0.005 and [K.p6_delta(d, b) for b in K.P6_BANDS] == [0.010, 0.015, 0.009]
    assert not any(K.p6_khong_khung(d, b, ['C8-L-F02-2_lai1']) for b in K.P6_BANDS)
    # never over an existing file
    assert M.main(['--p6', ps[0], '--session-nhan', ses, '--nguong', ng, '--quyet-dinh', 'x', '--out', out]) == 2


def test_cli_refuses(tmp_path, capsys):
    ps, ses, ng, out, d = _made(tmp_path)
    base = ['--p6', ps[0], '--p6', ps[1], '--nguong', ng]
    assert M.main(base + ['--session-nhan', ses, '--quyet-dinh', '  ', '--out', str(tmp_path / 'o1.json')]) == 2
    other = tmp_path / 'nguong2.json'
    other.write_text(json.dumps(dict(x=2)), encoding='utf-8')
    assert M.main(['--p6', ps[0], '--nguong', str(other), '--session-nhan', ses, '--quyet-dinh', 'q',
                   '--out', str(tmp_path / 'o2.json')]) == 2
    assert M.main(base + ['--session-nhan', ses, '--session-nhan', ses, '--quyet-dinh', 'q',
                          '--out', str(tmp_path / 'o3.json')]) == 2
    # the receiving session is the session of a source: it has a P6 of its own
    a = json.loads(open(ps[0], encoding='utf-8').read())
    a['nguon'][0]['sha256'] = cu.sha256(ses)
    open(ps[0], 'w', encoding='utf-8').write(json.dumps(a))
    assert M.main(base + ['--session-nhan', ses, '--quyet-dinh', 'q', '--out', str(tmp_path / 'o4.json')]) == 2
    assert not any(os.path.exists(tmp_path / f'o{i}.json') for i in (1, 2, 3, 4))
    assert capsys.readouterr().out.count('#ket_qua,ma=2') == 4


def test_kiem_accepts_only_what_it_can_rederive(tmp_path):
    ps, ses, ng, out, d = _made(tmp_path)
    s, n = {cu.sha256(ses)}, {cu.sha256(ng)}
    assert M.kiem(d, s, n)['quyet_dinh'] == 'L172 Liam 25/11'

    def bad(mut, s_=s, n_=n):
        x = copy.deepcopy(d)
        mut(x)
        with pytest.raises(M.MuonError):
            M.kiem(x, s_, n_)
    bad(lambda x: None, {'9' * 64})                                    # made for another session
    bad(lambda x: None, s | {'9' * 64})                                # the lock has one more session
    bad(lambda x: None, set())                                         # the lock names no session
    bad(lambda x: None, s, {'9' * 64})                                 # another nguong file
    bad(lambda x: None, s, set())
    bad(lambda x: x['muon'].update(version='c8b_p6_muon_v0'))
    bad(lambda x: x['muon'].update(quy_tac='trung_binh'))
    bad(lambda x: x['muon'].update(quyet_dinh=' '))
    bad(lambda x: x.update(muon=True))
    bad(lambda x: x.update(luot=[dict(luot='C8-L-F02-2_lai1')]))
    bad(lambda x: x['nguon'].append(dict(vai='session', sha256=cu.sha256(ses))))
    bad(lambda x: x['dai']['B2'].update(delta_h_m=0.011))               # a smaller number than the largest
    bad(lambda x: x['dai']['B3'].update(delta_h_m=0.02))
    bad(lambda x: x['dai']['B0'].update(muon_tu_sha256='0' * 64))
    bad(lambda x: x['muon']['nguon_p6'].pop())                         # a source dropped: dai no longer follows
    bad(lambda x: x['muon'].update(nguon_p6=[]))
    bad(lambda x: x['muon']['nguon_p6'][0].update(sha256='0' * 64))
    bad(lambda x: x['muon']['nguon_p6'][0].update(duong_dan=str(tmp_path / 'khong_co.json')))
    bad(lambda x: x['muon']['nguon_p6'][1]['dai']['B3'].update(trang_thai='dat', delta_h_m=0.001))
    bad(lambda x: x['muon']['session_nhan_sha256'].append(cu.sha256(ses)))
    bad(lambda x: [r for r in x['nguon'] if r['vai'] == 'nguong'][0].update(sha256='9' * 64))
    # a source of the session of the lock itself
    bad(lambda x: x['muon'].update(session_nhan_sha256=['1' * 64]), {'1' * 64})
    # a source file edited after the borrowing
    a = json.loads(open(ps[0], encoding='utf-8').read())
    a['dai']['B1']['delta_h_m'] = 0.02
    open(ps[0], 'w', encoding='utf-8').write(json.dumps(a))
    with pytest.raises(M.MuonError, match='changed since'):
        M.kiem(d, s, n)


def _a6():
    return dict(h_goc_m=[0.123, 0.122, 0.125, 0.126], u_h_goc_m=[0.013] * 4, H_m=2.632, h_m=0.124175,
                mien=dict(rho_max_m=1.615))


def _tb(level):
    return dict(version='c8b_3b_v2', task_rates=level, dung_duoc=True, ga_ok=False, b_dung_ke_ca_troi=True,
                d_p_quay_m=0.002213, d_psi_quay_deg=0.519, thieu=[], gioi_han=['x'],
                B=dict(shutter=dict(use_alpha=False, alpha_r_le_s_px=5e-6, alpha_c_le_s_px=-1e-6, mot_khung_s=1 / 60,
                                    why='spread')))


def test_lock_uses_the_borrowed_numbers_and_says_so(tmp_path):
    ps, ses, ng, out, d = _made(tmp_path)
    tbs = {lv: _tb(lv) for lv in K.LEVELS}
    k = K.build(tbs, a_m=0.1445, a6=_a6(), p6_dong=d)
    plain = dict(schema=K.P6_VERSION, dai={b: dict(delta_h_m=d['dai'][b]['delta_h_m'], trang_thai='dat')
                                           for b in M.BANDS})
    k0 = K.build(tbs, a_m=0.1445, a6=_a6(), p6_dong=plain)
    c6 = K.p6_he_so(_a6())
    for lv in K.LEVELS:
        assert k['muc'][lv]['hang_theo_u'] == k0['muc'][lv]['hang_theo_u']          # the same bounds
        note = k['muc'][lv]['hang_theo_u_ly_do']
        assert 'MUON' in note and 'L172 Liam 25/11' in note and M.VERSION in note and '2 file' in note
        assert all(r['sha256'][:12] in note for r in d['muon']['nguon_p6'])
        assert k0['muc'][lv]['hang_theo_u_ly_do'] is None                          # a computed file: no note
    b8 = k['muc']['O1-F08']['hang_theo_u']
    cq = k['muc']['O1-F08']['can_quay']
    assert [b['P6_rieng_ii_m'] for b in b8] == pytest.approx([c6 * 0.010, c6 * 0.015, c6 * 0.009], rel=1e-12)
    assert b8[1]['P4_m'] == pytest.approx(cq['p_m'] + K.LE * c6 * 0.015, rel=1e-12)
    assert (k['trang_thai'], k['ly_do_chua_khoa']) == (k0['trang_thai'], k0['ly_do_chua_khoa'])   # the note stops nothing
    # a borrowed band nobody has stays thieu with the usual reason next to the note
    d3 = copy.deepcopy(d)
    d3['dai']['B3'] = dict(delta_h_m=None, trang_thai='thieu', n_nguon=0, muon_tu_sha256=None)
    k3 = K.build(tbs, a_m=0.1445, a6=_a6(), p6_dong=d3)
    assert k3['muc']['O1-F08']['hang_theo_u'][2]['trang_thai'] == 'thieu'
    assert not k3['muc']['O1-F08']['hang_theo_u'][2]['khong_khung']
    assert 'B3: thieu delta_h_dong' in k3['muc']['O1-F08']['hang_theo_u_ly_do'] and 'MUON' in k3['muc']['O1-F08']['hang_theo_u_ly_do']


def test_lock_cli_takes_it_only_for_its_own_session(tmp_path, capsys):
    ps, ses, ng, out, d = _made(tmp_path)
    tb = tmp_path / '3b.json'
    tb.write_text(json.dumps(_tb('O1-F05')), encoding='utf-8')
    base = ['--tb', f'O1-F05={tb}', '--canh-ma-m', '0.1445', '--muc', 'O1-F05', '--nguong', ng, '--p6-dong', out]
    ko = tmp_path / 'k.json'
    assert K.main(base + ['--session', ses, '--out', str(ko)]) == 0
    k = json.loads(ko.read_text(encoding='utf-8'))
    assert 'MUON' in k['muc']['O1-F05']['hang_theo_u_ly_do']
    assert any(n['vai'] == 'p6_dong' and n['sha256'] == cu.sha256(out) for n in k['khoa_truoc']['nguon'])
    capsys.readouterr()
    # another session, no session, a session more: refused, nothing written
    ses2 = tmp_path / 'session2.json'
    ses2.write_text(json.dumps(dict(buoi='khac')), encoding='utf-8')
    for i, extra in enumerate((['--session', str(ses2)], [], ['--session', ses, '--session', str(ses2)])):
        o = tmp_path / f'k{i}.json'
        assert K.main(base + extra + ['--out', str(o)]) == 2 and not o.exists()
        assert 'P6 muon' in capsys.readouterr().err
    # a computed P6 file (no block muon) goes on as before, without the check and without the note
    plain = tmp_path / 'p6.json'
    plain.write_text(json.dumps(_p6(cu.sha256(str(ses2)), dict(B0=0.004, B1=0.010, B2=0.011, B3=0.009),
                                    ng=cu.sha256(ng))), encoding='utf-8')
    o = tmp_path / 'kp.json'
    assert K.main(base[:-1] + [str(plain), '--session', str(ses2), '--out', str(o)]) == 0
    assert 'MUON' not in json.loads(o.read_text(encoding='utf-8'))['muc']['O1-F05']['hang_theo_u_ly_do']   # no --a6: thieu
