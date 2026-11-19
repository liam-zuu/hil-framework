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


def xoay_file(tmp_path, q=(0.03, -0.01), st='dat'):
    p = tmp_path / 'xoay.json'
    p.write_text(json.dumps(dict(schema='xoay_v1', q_tag_m=list(q), u_q_m=0.0005, trang_thai=st)), encoding='utf-8')
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


def test_another_mounting_waits_for_the_c8b_composition(fake_a6, tmp_path):
    a6p, _ = fake_a6
    ll = dict(schema='lan_lap_v1', tam_id='T1', cung_lan_lap=False, trang_thai={'delta_c_B': 'dat'})
    out = ck.chuoi_khung(spin_rows(), ck.doc_a6(a6p), xoay_file(tmp_path), lan_lap=ll, track=TRACK0)
    assert all(r['a6_ok'] and not r['q_ok'] and 'cho_ham_ghep_q_C8B' in r['ly_do'] for r in out)
    ll['trang_thai'] = {'delta_c_B': 'chua_ket_luan'}
    out = ck.chuoi_khung(spin_rows(), ck.doc_a6(a6p), xoay_file(tmp_path), lan_lap=ll, track=TRACK0)
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
