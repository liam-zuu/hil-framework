"""Tests for tools/c7_chuoi_khung.py (F1 c7_chuoi_khung_v1) on SYNTHETIC frames built here (not a measurement). The
A6 module of CODE_C8B (c8b_a6, contract section 2.C) is not merged yet: a fake module with the agreed signature
(load_a6, ap_dung, delta_psi_a6) stands in for it."""
import json
import math
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import c7_chuoi_khung as ck  # noqa: E402

TRACK0 = dict(tag_offset=[0.0, 0.0, 0.0])
G_B = (0.02, -0.01)            # fake plate tilt (dh/dx in the tag frame)
H_M, h_M = ck.ZC_16_11, 0.124


def _fake_dpsi(xy, psi, plumb):
    """[R(psi) g_b] x (t - C) / (2 (H - h)): the A6 formula of the contract, used to build and to undo."""
    gx = np.cos(psi) * G_B[0] - np.sin(psi) * G_B[1]
    gy = np.sin(psi) * G_B[0] + np.cos(psi) * G_B[1]
    dx, dy = xy[:, 0] - plumb[0], xy[:, 1] - plumb[1]
    return (gx * dy - gy * dx) / (2.0 * (H_M - h_M))


@pytest.fixture
def fake_a6(monkeypatch, tmp_path):
    calls = dict(n=0)
    mod = types.ModuleType('c8b_a6')

    def load_a6(path):
        d = json.load(open(path, encoding='utf-8'))
        if d.get('schema') != 'c8b_a6_v2':
            raise ValueError('schema')
        return d

    def ap_dung(a6, tam_id):
        if tam_id == a6['tam_id'] or (tam_id is None and a6.get('cho_none')):
            return 'dat', ''
        return 'chua_ket_luan', f'tam {tam_id} khac {a6["tam_id"]}'

    def delta_psi_a6(a6, xy, psi, plumb, zc):
        calls['n'] += 1
        if abs(zc - H_M) > 1e-9 or abs(plumb[0] - ck.PLUMB_16_11[0]) > 1e-9:
            raise ValueError('C, H differ from the file')
        xy, psi = np.asarray(xy, float), np.asarray(psi, float)
        d = _fake_dpsi(xy, psi, plumb)
        rho = np.hypot(xy[:, 0] - plumb[0], xy[:, 1] - plumb[1])
        ok = rho <= a6['rho_max']
        return d, ok, np.where(ok, '', 'ngoai_mien')
    mod.load_a6, mod.ap_dung, mod.delta_psi_a6 = load_a6, ap_dung, delta_psi_a6
    monkeypatch.setitem(sys.modules, 'c8b_a6', mod)
    p = tmp_path / 'a6.json'
    p.write_text(json.dumps(dict(schema='c8b_a6_v2', tam_id='T1', rho_max=1.6, cho_none=True)), encoding='utf-8')
    return str(p), calls


def xoay_file(tmp_path, q=(0.03, -0.01), st='dat', a6_sha='auto'):
    """xoay_v1 fitted on psi' after the A6 file of tmp_path (a6_sha256 of it, contract F4); a6_sha None: no key."""
    p = tmp_path / 'xoay.json'
    d = dict(schema='xoay_v1', q_tag_m=list(q), u_q_m=0.0005, trang_thai=st)
    if a6_sha == 'auto' and (tmp_path / 'a6.json').exists():
        a6_sha = ck.sha256(str(tmp_path / 'a6.json'))
    if a6_sha not in (None, 'auto'):
        d['a6_sha256'] = a6_sha
    p.write_text(json.dumps(d), encoding='utf-8')
    return str(p)


def spin_rows(c=(0.4, -0.2), q=(0.03, -0.01), n=240, valid=True):
    """Car spinning about its centre c: tag centre t = c + R(psi) q, measured heading = psi + dpsi_A6 (the error the
    chain removes)."""
    psi = np.linspace(-3.0, 3.0, n)
    rows = []
    for k, p in enumerate(psi):
        tx = c[0] + math.cos(p) * q[0] - math.sin(p) * q[1]
        ty = c[1] + math.sin(p) * q[0] + math.cos(p) * q[1]
        d = float(_fake_dpsi(np.array([[tx, ty]]), np.array([p]), ck.PLUMB_16_11)[0])
        rows.append(dict(idx=k, t_s=k / 60.0, valid=valid, car_seen=valid, tag_x=tx, tag_y=ty,
                         psi=float(ck.wrap(p + d)), true_psi=p))
    return rows


def test_centre_recovered_after_a6_and_q(fake_a6, tmp_path):
    a6p, calls = fake_a6
    rows = spin_rows()
    out = ck.chuoi_khung(rows, ck.doc_a6(a6p), xoay_file(tmp_path), track=TRACK0)
    assert calls['n'] == 1
    assert all(r['trang_thai'] == 'khoa' and r['ly_do'] == '' for r in out)
    px, py = np.array([r['p_x'] for r in out]), np.array([r['p_y'] for r in out])
    assert np.max(np.hypot(px - 0.4, py + 0.2)) < 1e-6
    # psi' is the true heading back (A6 removed with the measured heading, a small second-order difference)
    err = np.abs(ck.wrap(np.array([r['psi_ma_p'] for r in out]) - np.array([r['true_psi'] for r in rows])))
    assert err.max() < 1e-4
    assert set(out[0]) == set(ck.ROW_KEYS)


def test_without_a6_every_frame_is_diagnostic(tmp_path):
    out = ck.chuoi_khung(spin_rows(), None, xoay_file(tmp_path), track=TRACK0)
    assert all(r['trang_thai'] == 'chan_doan' and 'thieu_a6' in r['ly_do'] for r in out)
    assert all(math.isnan(r['psi_ma_p']) and math.isnan(r['p_x']) for r in out)


def test_other_plate_refuses_a6(fake_a6, tmp_path):
    a6p, _ = fake_a6
    d = json.load(open(a6p))
    d['cho_none'] = False
    open(a6p, 'w').write(json.dumps(d))
    ll = dict(schema='lan_lap_v1', tam_id='T2', cung_lan_lap=True, trang_thai={'delta_c_B': 'dat'})
    out = ck.chuoi_khung(spin_rows(), ck.doc_a6(a6p), xoay_file(tmp_path), lan_lap=ll, track=TRACK0)
    assert all(not r['a6_ok'] and r['trang_thai'] == 'chan_doan' and r['ly_do'].startswith('a6_chua_ket_luan')
               for r in out)


def test_another_mounting_uses_the_c8b_composition(fake_a6, tmp_path, monkeypatch):
    a6p, _ = fake_a6
    ll = dict(schema='lan_lap_v1', tam_id='T1', cung_lan_lap=False, trang_thai={'delta_c_B': 'dat'})
    out = ck.chuoi_khung(spin_rows(), ck.doc_a6(a6p), xoay_file(tmp_path), lan_lap=ll, track=TRACK0)
    assert all(r['a6_ok'] and not r['q_ok'] and 'thieu_cho_cho_ghep_q' in r['ly_do'] for r in out)
    seen = {}
    mod = types.ModuleType('c8b_xoay')

    def ghep_q(lan_lap, cho, xoay, truc_xe=None, u_psi_w_rad=None):     # F16 (C8B, hil 6866824)
        seen.update(lan_lap=lan_lap, cho=cho, xoay=xoay, truc_xe=truc_xe, u_w=u_psi_w_rad)
        return dict(q_m=[0.02, 0.015], u_q_m=0.0007, trang_thai=lan_lap.get('st_thu', 'dat'), ly_do='')
    mod.ghep_q = ghep_q
    monkeypatch.setitem(sys.modules, 'c8b_xoay', mod)
    rows = spin_rows(q=(0.02, 0.015))
    out = ck.chuoi_khung(rows, ck.doc_a6(a6p), xoay_file(tmp_path), lan_lap=ll, track=TRACK0, cho='K1',
                         truc_xe=dict(schema='truc_xe_v1'), u_psi_w_rad=0.001)
    assert seen['cho'] == 'K1' and all(r['q_ok'] and r['u_q_m'] == 0.0007 for r in out)
    assert seen['xoay']['q_tag_m'] == [0.03, -0.01] and seen['truc_xe'] == dict(schema='truc_xe_v1')
    assert seen['u_w'] == 0.001
    px, py = np.array([r['p_x'] for r in out]), np.array([r['p_y'] for r in out])
    assert np.max(np.hypot(px - 0.4, py + 0.2)) < 1e-6
    ll['st_thu'] = 'chua_ket_luan'
    out = ck.chuoi_khung(rows, ck.doc_a6(a6p), xoay_file(tmp_path), lan_lap=ll, track=TRACK0, cho='K1')
    assert all(not r['q_ok'] and 'ghep_q_chua_ket_luan' in r['ly_do'] for r in out)
    ll['trang_thai'] = {'delta_c_B': 'chua_ket_luan'}
    out = ck.chuoi_khung(rows, ck.doc_a6(a6p), xoay_file(tmp_path), lan_lap=ll, track=TRACK0, cho='K1')
    assert all('thieu_delta_c_B' in r['ly_do'] for r in out)


def test_q_not_dat_or_missing(fake_a6, tmp_path):
    a6p, _ = fake_a6
    out = ck.chuoi_khung(spin_rows(), ck.doc_a6(a6p), xoay_file(tmp_path, st='chua_ket_luan'), track=TRACK0)
    assert all(not r['q_ok'] and 'xoay_trang_thai_chua_ket_luan' in r['ly_do'] for r in out)
    out = ck.chuoi_khung(spin_rows(), ck.doc_a6(a6p), None, track=TRACK0)
    assert all(not r['q_ok'] and 'thieu_xoay' in r['ly_do'] for r in out)


def test_frame_gate_and_a6_range(fake_a6, tmp_path):
    a6p, _ = fake_a6
    rows = spin_rows()
    rows[5]['valid'] = False
    rows[6]['tag_x'] = 5.0                                   # outside the A6 range rho_max
    out = ck.chuoi_khung(rows, ck.doc_a6(a6p), xoay_file(tmp_path), track=TRACK0)
    assert not out[5]['cong_ok'] and out[5]['ly_do'].startswith('cong_khung')
    assert out[6]['cong_ok'] and not out[6]['a6_ok'] and 'ngoai_mien' in out[6]['ly_do']
    assert sum(r['trang_thai'] == 'khoa' for r in out) == len(out) - 2


def test_tag_heading_source_checked(fake_a6, tmp_path):
    with pytest.raises(ValueError, match='tag_offset'):
        ck.chuoi_khung(spin_rows(), None, None)
    with pytest.raises(ValueError, match='tag_offset'):
        ck.chuoi_khung(spin_rows(), None, None, track=dict(tag_offset=[0.01, 0.0, 0.0]))
    rows = spin_rows()
    for r in rows:
        r['tag_psi'], r['psi'] = r['psi'], 99.0              # the tag_psi column wins when present
    a6p, _ = fake_a6
    out = ck.chuoi_khung(rows, ck.doc_a6(a6p), xoay_file(tmp_path))
    assert all(r['trang_thai'] == 'khoa' for r in out)


def test_wrong_c_h_refused(fake_a6, tmp_path):
    a6p, _ = fake_a6
    with pytest.raises(ValueError, match='C, H'):
        ck.chuoi_khung(spin_rows(), ck.doc_a6(a6p), xoay_file(tmp_path), zc=2.5, track=TRACK0)


def test_mean_unwrapped_across_pi():
    a = np.array([math.pi - 0.01, -math.pi + 0.01, math.pi - 0.02, -math.pi + 0.02])
    m = ck.trung_binh_go_vong(a)
    assert abs(ck.wrap(m - math.pi)) < 1e-12
    b = a + 0.3
    assert abs(ck.wrap(ck.trung_binh_go_vong(b) - m) - 0.3) < 1e-12        # mean of differences
    assert math.isnan(ck.trung_binh_go_vong([float('nan')]))


def test_wrap_half_open():
    assert ck.wrap(-math.pi) == math.pi and ck.wrap(math.pi) == math.pi
    assert abs(ck.wrap(3 * math.pi / 2) + math.pi / 2) < 1e-12


def test_q_needs_the_a6_it_was_fitted_on(fake_a6, tmp_path):
    """Codex CT1-3: q of xoay_v1 is fitted on psi' after ONE A6 (a6_sha256, contract F4): another A6, or no a6_sha256,
    gives no centre."""
    a6p, _ = fake_a6
    for sha, why in ((None, 'xoay_thieu_a6_sha256'), ('0' * 64, 'xoay_a6_khac_a6_dang_dung')):
        out = ck.chuoi_khung(spin_rows(), ck.doc_a6(a6p), xoay_file(tmp_path, a6_sha=sha), track=TRACK0)
        assert not any(r['q_ok'] for r in out) and all(why in r['ly_do'] for r in out)
        assert all(r['a6_ok'] for r in out)                      # the heading after A6 stays


def test_blank_tag_psi_cells_are_missing_frames(fake_a6, tmp_path):
    """Codex CT1-18: a tag_psi column (text in frames.csv) with blank cells at the start and inside: those frames have
    no tag heading, the others keep their IDs and values; the column is used even when its first cell is blank."""
    a6p, _ = fake_a6
    rows = spin_rows()
    for r in rows:
        r['tag_psi'], r['psi'] = str(r['psi']), 99.0
    rows[0]['tag_psi'] = ''
    rows[50]['tag_psi'] = ''
    out = ck.chuoi_khung(rows, ck.doc_a6(a6p), xoay_file(tmp_path))
    assert [r['frame_id'] for r in out] == [int(r['idx']) for r in rows]
    assert not out[0]['cong_ok'] and not out[50]['cong_ok'] and out[1]['cong_ok'] and out[51]['cong_ok']
    assert abs(out[1]['tag_psi'] - float(rows[1]['tag_psi'])) < 1e-12


def test_lan_lap_session_file_gives_the_run(tmp_path):
    """Codex CT1-3: a lan_lap_v1 file of the session (c7_nhac_tam: file keys + 'luot') gives the run of luot_id with
    the file keys copied in; several runs without luot_id: refused."""
    d = dict(schema='lan_lap_v1', cua_so_v1_sha256='a' * 64, session_sha256='b' * 64, tam_id='T1',
             luot=[dict(schema='lan_lap_v1', luot_id='L_1', cho='goc'), dict(schema='lan_lap_v1', luot_id='L_2')])
    p = tmp_path / 'lan_lap.json'
    p.write_text(json.dumps(d), encoding='utf-8')
    r = ck.doc_lan_lap(str(p), 'L_1')
    assert r['luot_id'] == 'L_1' and r['cua_so_v1_sha256'] == 'a' * 64 and r['_sha256'] == ck.sha256(str(p))
    with pytest.raises(ValueError, match='2 runs'):
        ck.doc_lan_lap(str(p))
    with pytest.raises(ValueError, match='L_9'):
        ck.doc_lan_lap(str(p), 'L_9')


def test_xoay_per_place_file(fake_a6, tmp_path):
    """xoay_v1 per place (C8B ghep_q, hil 6866824: 'version', 'cho' {place: {q_tag_m, u_q_m, trang_thai}}): the q of
    the place of the run; another place missing: no q."""
    a6p, _ = fake_a6
    p = tmp_path / 'xoay_cho.json'
    p.write_text(json.dumps(dict(version='xoay_v1', a6_sha256=ck.sha256(a6p), cho={'goc': dict(
        q_tag_m=[0.03, -0.01], u_q_m=0.0005, trang_thai='dat')})), encoding='utf-8')
    out = ck.chuoi_khung(spin_rows(), ck.doc_a6(a6p), str(p), track=TRACK0, cho='goc')
    assert all(r['q_ok'] for r in out)
    out = ck.chuoi_khung(spin_rows(), ck.doc_a6(a6p), str(p), track=TRACK0, cho='K1')
    assert all(not r['q_ok'] and 'xoay_thieu_cho_K1' in r['ly_do'] for r in out)


def _xoay_goc_k1(tmp_path, a6p, k1=True, k1_st='dat'):
    cho = {'goc': dict(q_tag_m=[0.03, -0.01], u_q_m=0.0005, trang_thai='dat')}
    if k1:
        cho['K1'] = dict(q_tag_m=[0.02, 0.01], u_q_m=0.0004, trang_thai=k1_st)
    p = tmp_path / 'xoay_goc_k1.json'
    p.write_text(json.dumps(dict(version='xoay_v1', a6_sha256=ck.sha256(a6p), cho=cho)), encoding='utf-8')
    return str(p)


def test_k2_takes_the_block_of_a_place_with_e_on_the_same_mounting(fake_a6, tmp_path):
    """CODE 22/11 option (A) (contract C8 section 3, "F4 cho K2"): K2 has no F4 block (no E turn); it takes the q of
    the place with E (K1 here) only when the lan_lap_id of the two F5 (F8 run 1) are equal and given."""
    a6p, _ = fake_a6
    xp = _xoay_goc_k1(tmp_path, a6p)
    rows = spin_rows(q=(0.02, 0.01))
    out = ck.chuoi_khung(rows, ck.doc_a6(a6p), xp, track=TRACK0, cho='K2', cho_q='K1', ll_id='LL_7', ll_id_q='LL_7')
    assert all(r['q_ok'] and r['trang_thai'] == 'khoa' for r in out)
    px, py = np.array([r['p_x'] for r in out]), np.array([r['p_y'] for r in out])
    assert np.max(np.hypot(px - 0.4, py + 0.2)) < 1e-6                 # the K1 q recovers the centre
    assert all(r['u_q_m'] == 0.0004 for r in out)
    xo = ck.doc_xoay_cung_lan_lap(xp, 'K2', 'K1', 'LL_7', 'LL_7')
    assert xo['ok'] and xo['nguon_q'] == dict(cho='K1', lan_lap_id='LL_7',
                                              cach='khoi F4 cua cho co E cung lan lap tam (CODE 22/11 A)')


@pytest.mark.parametrize('ll, llq, why', [('LL_7', 'LL_8', 'khac_lan_lap_K1'), (None, 'LL_7', 'chua_xac_minh'),
                                          ('LL_7', None, 'chua_xac_minh'), ('', '', 'chua_xac_minh')])
def test_k2_other_or_unverified_mounting_has_no_q(fake_a6, tmp_path, ll, llq, why):
    a6p, _ = fake_a6
    out = ck.chuoi_khung(spin_rows(), ck.doc_a6(a6p), _xoay_goc_k1(tmp_path, a6p), track=TRACK0, cho='K2',
                         cho_q='K1', ll_id=ll, ll_id_q=llq)
    assert all(not r['q_ok'] and r['u_q_m'] is None and r['trang_thai'] == 'chan_doan' and why in r['ly_do']
               for r in out)


def test_k2_rules_the_own_block_wins_and_bad_sources_refused(fake_a6, tmp_path):
    a6p, _ = fake_a6
    xp = _xoay_goc_k1(tmp_path, a6p)
    # a place with its own block uses it, whatever cho_q says
    xo = ck.doc_xoay_cung_lan_lap(xp, 'goc', 'K1', 'LL_1', 'LL_2')
    assert xo['ok'] and xo['q'] == [0.03, -0.01] and xo['nguon_q']['cho'] == 'goc'
    # without cho_q: as before (no q)
    assert ck.doc_xoay_cung_lan_lap(xp, 'K2')['ly_do'] == 'xoay_thieu_cho_K2'
    with pytest.raises(ValueError):
        ck.doc_xoay_cung_lan_lap(xp, 'K2', 'K2', 'LL_7', 'LL_7')
    # the place with E has no dat block: no q
    xp2 = _xoay_goc_k1(tmp_path, a6p, k1_st='chua_ket_luan')
    xo = ck.doc_xoay_cung_lan_lap(xp2, 'K2', 'K1', 'LL_7', 'LL_7')
    assert not xo['ok'] and xo['q'] is None and xo['ly_do'].startswith('cho_K1_cung_lan_lap:')
    xp3 = _xoay_goc_k1(tmp_path, a6p, k1=False)
    assert ck.doc_xoay_cung_lan_lap(xp3, 'K2', 'K1', 'LL_7', 'LL_7')['ly_do'].startswith('cho_K1_cung_lan_lap:')


# ---------------------------------------------------------------- tam_id= (CODE 20/11, HIEUCHUAN HCV3-1)
def _a6_needs_plate(a6p):
    d = json.load(open(a6p))
    d['cho_none'] = False                     # ap_dung(a6, None) -> chua_ket_luan, as C8B does
    open(a6p, 'w').write(json.dumps(d))


def test_tam_cua_luot_rules():
    assert ck.tam_cua_luot(None) == (None, '')
    assert ck.tam_cua_luot(None, 'T1') == ('T1', '')
    assert ck.tam_cua_luot(dict(tam_id='T1')) == ('T1', '')
    assert ck.tam_cua_luot(dict(tam_id='T1'), 'T1') == ('T1', '')
    assert ck.tam_cua_luot(dict(), 'T1') == ('T1', '')                 # lan_lap without tam_id
    assert ck.tam_cua_luot(dict(tam_id='T1'), 'T2') == (None, 'a6_tam_id_khac_lan_lap')


def test_tam_id_without_lan_lap_applies_a6(fake_a6, tmp_path):
    a6p, calls = fake_a6
    _a6_needs_plate(a6p)
    out = ck.chuoi_khung(spin_rows(), ck.doc_a6(a6p), xoay_file(tmp_path), track=TRACK0)
    assert all(not r['a6_ok'] and r['ly_do'].startswith('a6_chua_ket_luan') for r in out)   # the HCV3-1 case
    assert calls['n'] == 0
    out = ck.chuoi_khung(spin_rows(), ck.doc_a6(a6p), xoay_file(tmp_path), track=TRACK0, tam_id='T1')
    assert calls['n'] == 1
    assert all(r['trang_thai'] == 'khoa' and r['ly_do'] == '' for r in out)
    out = ck.chuoi_khung(spin_rows(), ck.doc_a6(a6p), xoay_file(tmp_path), track=TRACK0, tam_id='T2')
    assert all(not r['a6_ok'] and r['ly_do'].startswith('a6_chua_ket_luan') for r in out)


def test_tam_id_and_lan_lap(fake_a6, tmp_path):
    a6p, calls = fake_a6
    _a6_needs_plate(a6p)
    ll = dict(schema='lan_lap_v1', tam_id='T1', cung_lan_lap=True, trang_thai={'delta_c_B': 'dat'})
    out = ck.chuoi_khung(spin_rows(), ck.doc_a6(a6p), xoay_file(tmp_path), lan_lap=ll, track=TRACK0, tam_id='T1')
    assert all(r['trang_thai'] == 'khoa' for r in out)
    n = calls['n']
    out = ck.chuoi_khung(spin_rows(), ck.doc_a6(a6p), xoay_file(tmp_path), lan_lap=ll, track=TRACK0, tam_id='T2')
    assert calls['n'] == n                    # A6 not applied
    assert all(not r['a6_ok'] and r['trang_thai'] == 'chan_doan' and 'a6_tam_id_khac_lan_lap' in r['ly_do']
               and math.isnan(r['psi_ma_p']) for r in out)
    ll2 = dict(schema='lan_lap_v1', cung_lan_lap=True, trang_thai={'delta_c_B': 'dat'})          # no tam_id
    out = ck.chuoi_khung(spin_rows(), ck.doc_a6(a6p), xoay_file(tmp_path), lan_lap=ll2, track=TRACK0, tam_id='T1')
    assert all(r['trang_thai'] == 'khoa' for r in out)
