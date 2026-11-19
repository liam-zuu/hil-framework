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
    k = K.build({'O1-F02': _tb('O1-F02', alpha=True), 'O1-F05': _tb('O1-F05')})
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
