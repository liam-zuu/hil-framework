"""tools/gate_c8.py: the C8 gate file gate_c8_v2 (check, check_nguon, Uhat, mask, chains, evidence, premise).
CODE_GATE 19/11/2026, design CLAUDE_THESIS 03_CODE/docs/2026-11-19_Thiet_ke_rq3_tuong_doi_v6_C8.md section 12 (after
the scoring rules J v6; tests taken from the counter-examples of Codex, design v5_C8 section 9). Every number is
invented (FAKE)."""
import copy
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import gate_c8 as GC          # noqa: E402
import o1_gate                # noqa: E402
import rq3_log as rl          # noqa: E402
from o1_feasibility import Schedule    # noqa: E402
from o1_tasks import FIG8              # noqa: E402

LEVELS = ('O1-F02', 'O1-F05', 'O1-F08')


def _good():
    return GC.fake_gate(LEVELS, u_p_m=0.002, u_psi_deg=2.0, u_t_s=0.014)


def _write(tmp_path, d, name='gate_c8.json'):
    p = str(tmp_path / name)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(d, f)
    return p


def _blk(level='O1-F08', hang=0.002, he_so_r=0.0, u_phi=0.0, bands_p=(), hang_psi_deg=2.0, bands_psi=(), a_m=0.1445):
    """A level block as uhat_level wants it (bands parsed)."""
    e = dict(a_m=a_m, u_p=dict(hang_m=hang, he_so_r=he_so_r, u_phi_rad=u_phi, hang_theo_u=list(bands_p)),
             u_psi=dict(hang_rad=math.radians(hang_psi_deg), hang_theo_u=list(bands_psi)))
    return GC._level_terms(e, level)


def _brute(level, u_t, f, step=1e-4, a_m=0.1445):
    """max over t on a fine grid of f(r(t), sup V, sup W) with windows of grid points (a value the true sup reaches)."""
    sch = Schedule(FIG8[level])
    m = int(round(u_t / step))
    t = np.arange(-u_t - 2 * m * step, sch.Tcmd + u_t + 2 * m * step, step)
    v, w, u, r = GC.schedule_samples(sch, t, a_m)
    win = 2 * m + 1
    vs = GC._slide(v, win, np.maximum)
    ws = GC._slide(w, win, np.maximum)
    c = np.arange(m, len(t) - m)
    keep = t[c] >= 0
    return float(np.max(f(r[c][keep], vs[c - m][keep], ws[c - m][keep])))


# ── Uhat (design section 4) ──────────────────────────────────────────────────────────────────────────────────

def test_vectorised_schedule_is_schedule_tau():
    sch = Schedule(FIG8['O1-F05'])
    ts = np.linspace(-1.0, sch.Tcmd + 1.0, 4001)
    tau, td = GC._tau_vec(sch, ts)
    for t, a, b in zip(ts, tau, td):
        assert (a, b) == pytest.approx(sch.tau(t), abs=1e-12)
    v, w, _, _ = GC.schedule_samples(sch, ts, 0.1)
    for t, vv, ww in zip(ts[::50], v[::50], w[::50]):
        sx, sy, sw = sch.setpoint(t)
        assert vv == pytest.approx(math.hypot(sx, sy), abs=1e-12) and ww == pytest.approx(abs(sw), abs=1e-12)


@pytest.mark.parametrize('level', LEVELS)
def test_lipschitz_constants_bound_the_samples(level):
    sch = Schedule(FIG8[level])
    lip = GC.lipschitz(sch, 0.1445)
    t = np.arange(-0.1, sch.Tcmd + 0.1, 1e-5)
    v, w, u, r = GC.schedule_samples(sch, t, 0.1445)
    for x, k in ((v, 'L_V'), (w, 'L_W'), (u, 'L_u'), (r, 'L_r')):
        assert np.max(np.abs(np.diff(x))) / 1e-5 <= lip[k] * (1 + 1e-9)          # L_r is reached at tau = 0


@pytest.mark.parametrize('u_t', [0.02775, 0.027, 0.0105, 0.0002])
def test_rt4_1_cell_bound_covers_the_window_ends(u_t):
    """RT4-1: the upper bound of a cell is >= the true sup of V over every window of a time in the cell, the ends
    included (checked on a 1 us grid near t = 0, where V rises)."""
    sch = Schedule(FIG8['O1-F08'])
    blk = _blk(hang=0.0)
    tm = GC._Terms('g3', blk, u_t)
    lip = GC.lipschitz(sch, blk['a_m'])
    _, (ta, tb), res = GC._grid_bounds(sch, [tm], lip, blk['a_m'], u_t, -u_t, 0.05, 1e-3)
    f_up = res[0][0]
    for j in range(len(ta)):
        fine = np.arange(ta[j] - u_t, tb[j] + u_t + 1e-7, 1e-6)
        v = GC.schedule_samples(sch, fine, blk['a_m'])[0]
        assert f_up[j] >= u_t * v.max() - 1e-15


def test_rt1_8_f08_case_against_a_fine_grid():
    """Codex rounds 1, 4: U_t 27 ms, U_p 18 mm + 0.006 r: the sup on a 0.1 ms grid (reached) is <= Uhat, and Uhat is
    tight (within a fraction of a millimetre); Uhat_duoi <= Uhat."""
    u_t = 0.027
    uh = GC.uhat_level('O1-F08', _blk(hang=0.018, he_so_r=0.006), u_t)['g3']
    brute = _brute('O1-F08', u_t, lambda r, vs, ws: 0.018 + 0.006 * r + u_t * vs)
    assert uh['uhat_duoi'] <= brute + 1e-12 and brute <= uh['uhat'] and uh['uhat'] - brute < 2e-4
    g5 = GC.uhat_level('O1-F08', _blk(hang_psi_deg=1.22), u_t)['g5']
    brute5 = _brute('O1-F08', u_t, lambda r, vs, ws: np.degrees(math.radians(1.22) + u_t * ws))
    assert g5['uhat_duoi'] <= brute5 + 1e-9 and brute5 <= g5['uhat'] and g5['uhat'] - brute5 < 0.05


def _u_peak(level, a_m=0.1445):
    sch = Schedule(FIG8[level])
    t = np.arange(0.0, sch.Tcmd, 1e-5)
    return float(GC.schedule_samples(sch, t, a_m)[2].max())


def test_rt4_2_a_null_band_under_the_true_peak_of_u_is_missing():
    """RT4-2: a null band starting just below the true peak of u (above the 1 ms grid peak or not) gives Uhat null
    (thieu): the u-range of the window, not the grid points, meets the band."""
    up = _u_peak('O1-F08')
    for start in (up - 1e-9, up - 1e-4, up * 0.5):
        bands = [dict(u_lo=0.0, u_hi=start, gia_tri=0.001), dict(u_lo=start, u_hi=10.0, gia_tri=None)]
        uh = GC.uhat_level('O1-F08', _blk(bands_p=bands), 0.027)
        assert uh['g2']['trang_thai'] == uh['g3']['trang_thai'] == 'thieu' and uh['g3']['uhat'] is None
        assert uh['g5']['trang_thai'] != 'thieu'                       # the null band is in U_p only
    bands = [dict(u_lo=0.0, u_hi=up + 0.05, gia_tri=0.001), dict(u_lo=up + 0.05, u_hi=10.0, gia_tri=None)]
    uh = GC.uhat_level('O1-F08', _blk(bands_p=bands), 0.027)
    assert uh['g3']['trang_thai'] == 'quyet' and uh['g2']['uhat'] == pytest.approx(0.003)


def test_band_table_beyond_its_end_is_missing():
    up = _u_peak('O1-F08')
    uh = GC.uhat_level('O1-F08', _blk(bands_p=[dict(u_lo=0.0, u_hi=up * 0.9, gia_tri=0.001)]), 0.027)
    assert uh['g2']['trang_thai'] == 'thieu'


@pytest.mark.parametrize('bands, msg', [
    ([dict(u_lo=0.1, u_hi=1.0, gia_tri=0.0)], 'first at 0'),
    ([dict(u_lo=0.0, u_hi=1.0, gia_tri=0.0), dict(u_lo=1.1, u_hi=2.0, gia_tri=0.0)], 'no gap'),
    ([dict(u_lo=0.0, u_hi=1.0, gia_tri=0.0), dict(u_lo=0.9, u_hi=2.0, gia_tri=0.0)], 'no gap'),
    ([dict(u_lo=0.0, u_hi=1.0, gia_tri=0.0), dict(u_lo=1.0, u_hi=1.0, gia_tri=0.0)], 'increasing'),
    ([dict(u_lo=0.0, u_hi=1.0, gia_tri=-0.1)], '>= 0'),
    ([dict(u_lo=0.0, u_hi=1.0)], 'keys'),
])
def test_band_tables_refused(bands, msg):
    with pytest.raises(GC.GateFileError, match=msg):
        GC.check_bands(bands, 't')


def _straddle(level='O1-F05', u_t=0.02):
    """hang_m so that half the G3 threshold lies between Uhat_duoi and Uhat of the first grid."""
    first = GC.uhat_level(level, _blk(level, hang=0.0), u_t)['g3']
    assert first['uhat_duoi'] < first['uhat']
    return 0.5 * o1_gate.THRESHOLDS['g3'] - 0.5 * (first['uhat'] + first['uhat_duoi'])


def test_rt3_1_refinement_tells_the_half_threshold():
    hang = _straddle()
    uh = GC.uhat_level('O1-F05', _blk('O1-F05', hang=hang), 0.02)['g3']
    assert uh['trang_thai'] in ('quyet', 'chi_bao_cao') and uh['delta_cuoi_s'] < GC.DELTA0_S
    assert uh['uhat_duoi'] <= uh['uhat']
    assert (uh['uhat'] <= 0.05) if uh['trang_thai'] == 'quyet' else (uh['uhat_duoi'] > 0.05)


def test_rt3_1_budget_spent_is_bien_nguong_and_needs_liam():
    hang = _straddle()
    blk = _blk('O1-F05', hang=hang, hang_psi_deg=0.1)                # G5 decides here (not report-only)
    n0 = GC.uhat_level('O1-F05', _blk('O1-F05', hang=0.0), 0.02)['g3']['so_lan_tinh']   # the first grid alone
    uh = GC.uhat_level('O1-F05', blk, 0.02, budget=n0 + 1)
    assert uh['g3']['trang_thai'] == 'bien_nguong'
    with pytest.raises(GC.GateFileError, match='CAN LIAM QUYET'):
        GC.report_only_of('O1-F05', uh)
    q = dict(g3=dict(chon='chi_bao_cao', ngay='2026-11-20', phien='local_x', nguyen_van='FAKE'))
    assert GC.report_only_of('O1-F05', uh, q) == (('g3',), {'g3': ['quyet_liam']})
    q['g3']['chon'] = 'quyet'
    assert GC.report_only_of('O1-F05', uh, q) == ((), {})


def test_regression_constant_terms():
    uh = GC.uhat_level('O1-F05', _blk('O1-F05', hang=0.004), 1e-9)
    assert uh['g2']['uhat'] == pytest.approx(0.004, abs=1e-15) and uh['g2']['trang_thai'] == 'quyet'


def test_half_threshold_equal_still_decides():
    uh = GC.uhat_level('O1-F05', _blk('O1-F05', hang=0.05), 0.01)
    assert uh['g2']['uhat'] == 0.05 and uh['g2']['trang_thai'] == 'quyet'
    uh = GC.uhat_level('O1-F05', _blk('O1-F05', hang=0.05 + 1e-12), 0.01)
    assert uh['g2']['trang_thai'] == 'chi_bao_cao'


def test_missing_closed_term_is_thieu():
    blk = _blk('O1-F05')
    blk['u_p']['he_so_r'] = None
    uh = GC.uhat_level('O1-F05', blk, 0.01)
    assert uh['g2']['trang_thai'] == uh['g3']['trang_thai'] == 'thieu' and uh['g5']['trang_thai'] == 'quyet'
    assert GC.report_only_of('O1-F05', uh)[0] == ('g2', 'g3')


# ── mask (design section 5) ──────────────────────────────────────────────────────────────────────────────────

def test_g5_f08_always_report_only():
    d = GC.fake_gate(('O1-F08',), u_psi_deg=0.1, u_t_s=0.001)
    e = d['dieu_2b']['per_level']['O1-F08']
    assert e['uhat']['g5']['trang_thai'] == 'quyet' and e['report_only'] == ['g5']
    assert e['ly_do_report_only'] == {'g5': ['liam_g5_f08']}
    e['report_only'] = []
    with pytest.raises(GC.GateFileError, match='report_only'):
        GC.check(d)


def test_stored_uhat_must_be_recomputed():
    d = _good()
    d['dieu_2b']['per_level']['O1-F05']['uhat']['g3']['uhat'] *= 1.001
    with pytest.raises(GC.GateFileError, match='recomputed'):
        GC.check(d)
    d = _good()
    d['dieu_2b']['per_level']['O1-F05']['uhat']['g3']['so_lan_tinh'] += 1
    with pytest.raises(GC.GateFileError, match='recomputed'):
        GC.check(d)


# ── chains (design section 6) ────────────────────────────────────────────────────────────────────────────────

def _e(rid, kq='dat', lh=(), d1='dat', dl=None, gio=True):
    ref = dict(duong_dan=f'FAKE/{rid}.json', sha256=GC.FAKE_SHA)
    if dl is None:                       # the C8b rule (design v7 7a): a slip counts unless lost by a boot
        dl = d1 == 'truot' and 'thao_tac' not in lh
    return dict(run_id=rid, ket_qua=kq, loai_hong=list(lh), kiem_gio_d1=d1, dem_truot_kiem_gio=dl, file_cham=ref,
                file_gio=ref if gio else None, file_chung_cu=None if kq == 'hong' else ref)


def _slip(rid, op=False, dl=None):
    return _e(rid, 'hong', ['thao_tac', 'he_do'] if op else ['he_do'], 'truot', dl)


def test_chain_last_lost_run_not_at_cap_refused():
    with pytest.raises(GC.GateFileError, match='run it again'):
        GC.count_chain([_e('a'), _slip('b')])
    assert GC.count_chain([_slip('a'), _e('b')])[0] == 'co_luot'


def test_chain_four_time_slips_close_the_cell():
    st, n = GC.count_chain([_slip(c) for c in 'abcd'])
    assert st == 'dong_tran' and n['kiem_gio'] == 4
    with pytest.raises(GC.GateFileError, match='after the cell was closed'):
        GC.count_chain([_slip(c) for c in 'abcd'] + [_e('e')])


def test_chain_m2_operation_runs_do_not_count_unless_independent():
    with pytest.raises(GC.GateFileError, match='run it again'):      # M2: four boot + slip runs, none counted
        GC.count_chain([_slip(c, op=True) for c in 'abcd'])
    st, n = GC.count_chain([_slip(c, op=True, dl=True) for c in 'abcd'])
    assert st == 'dong_tran' and n['kiem_gio'] == 4


def test_chain_reference_window_counter_and_both_flags():
    def cw(rid):
        return _e(rid, 'hong', ['cua_so_tham_chieu', 'he_do'], 'truot')
    st, n = GC.count_chain([cw(c) for c in 'abcd'])
    assert st == 'dong_tran' and n == {'kiem_gio': 4, 'cua_so': 4}
    op = [_e(c, 'hong', ['thao_tac', 'cua_so_tham_chieu'], 'khong_kiem', gio=False) for c in 'wxyz']
    assert GC.count_chain(op + [_e('v')]) == ('co_luot', {'kiem_gio': 0, 'cua_so': 0})


@pytest.mark.parametrize('bad, msg', [
    (dict(ket_qua='hong', loai_hong=[]), 'loai_hong'),
    (dict(loai_hong=['he_do']), 'loai_hong'),
    (dict(kiem_gio_d1='truot'), 'not lost'),
    (dict(dem_truot_kiem_gio=True), 'dem_truot_kiem_gio true only'),
    (dict(ket_qua='hong', loai_hong=['he_do'], kiem_gio_d1='truot', dem_truot_kiem_gio=False), 'not counted'),
    (dict(kiem_gio_d1='x'), 'F11'),
    (dict(file_gio=None), 'file_gio null'),
])
def test_chain_entry_refusals(bad, msg):
    with pytest.raises(GC.GateFileError, match=msg):
        GC._chain_entry(dict(_e('a'), **bad), 'x')


def test_gate_dong_tran_cell_and_level_still_locked():
    """KT6-11: a cell closed at its cap closes that cell only; the level locks with another cell of the controller."""
    d = _good()
    extra = dict(o='O1-F05_PID_b', muc='O1-F05', controller='PID', trang_thai='dong_tran',
                 chuoi=[_slip(f'b{i}') for i in range(4)])
    d['o_c8'].append(extra)
    GC.check(d)
    extra['trang_thai'] = 'co_luot'
    with pytest.raises(GC.GateFileError, match='recounted'):
        GC.check(d)


def test_runs_rebuilt_from_cells():
    d = _good()
    d['runs_c8']['O1-F05'].pop()
    with pytest.raises(GC.GateFileError, match='no run left out'):
        GC.check(d)
    d = _good()
    d['runs_c8']['O1-F05'].append(dict(d['runs_c8']['O1-F05'][0], run_id='EXTRA'))
    with pytest.raises(GC.GateFileError, match='no run left out'):
        GC.check(d)
    d = _good()
    d['runs_c8']['O1-F05'][0]['h_s'] += 0.001
    with pytest.raises(GC.GateFileError, match='h_s \\+ u_max_s'):
        GC.check(d)
    d = _good()
    d['dieu_2b']['per_level']['O1-F05']['u_t_c8_s'] = 0.010
    with pytest.raises(GC.GateFileError, match='largest h_s'):
        GC.check(d)


# ── evidence and levels (design section 7) ────────────────────────────────────────────────────────────────────

def _set_ev(d, level, ctrl, **kw):
    for r in d['runs_c8'][level]:
        if r['controller'] == ctrl:
            r['chung_cu']['bo_so'][0].update(kw)


def _refresh(d):
    """Recompute the derived blocks (levels, evidence lists) as the writer would."""
    for k in d['muc_kha_thi']:
        d['muc_loai_e_do'][k] = GC.evidence_of(d['runs_c8'][k])['loai']
    loai = [k for k in d['muc_kha_thi'] if d['muc_loai_e_do'][k]]
    d['chung_cu_muc'] = {k: dict(e_do=GC.evidence_of(d['runs_c8'][k])['e_do'], g4=GC.evidence_of(d['runs_c8'][k])['g4'])
                         for k in d['muc_kha_thi'] if k not in loai}
    d['levels'] = [k for k in d['muc_kha_thi'] if k not in loai and k not in d['muc_bo_theo_liam']]
    d['dieu_2b']['per_level'] = {k: v for k, v in d['dieu_2b']['per_level'].items() if k in d['levels']}
    d['quyet_liam'] = {k: v for k, v in d['quyet_liam'].items() if k in d['levels']}
    return d


def test_i_one_run_excludes_the_level_before_the_other_controller():
    """J v6 4 (a): one PID run with the e_do lower bound above D at every node excludes the level, though LADRC has
    no bound (it is not waited for)."""
    d = _good()
    _set_ev(d, 'O1-F05', 'PID', e_do_lo_vuot_D_moi_nut=True)
    _set_ev(d, 'O1-F05', 'LADRC', e_do_xac_dinh_moi_nut=False, g4_can_huu_han_mien_hop_moi_nut=False)
    GC.check(_refresh(d))
    assert 'O1-F05' not in d['levels'] and d['muc_loai_e_do']['O1-F05'] == ['FAKE_O1-F05_PID']


def test_i_both_scorings_needed_and_final_scoring_only():
    d = _good()
    for r in d['runs_c8']['O1-F05']:
        if r['controller'] == 'PID':
            r['chung_cu']['bo_so'] = [dict(r['chung_cu']['bo_so'][0], e_do_lo_vuot_D_moi_nut=True),
                                      dict(r['chung_cu']['bo_so'][0], e_do_lo_vuot_D_moi_nut=False)]
    GC.check(_refresh(d))
    assert 'O1-F05' in d['levels']                                      # J3-1: above at one calibration only
    d = _good()
    _set_ev(d, 'O1-F05', 'PID', e_do_lo_vuot_D_moi_nut=True)
    for r in d['runs_c8']['O1-F05']:
        r['chung_cu']['ban_cham_cuoi'] = False
    with pytest.raises(GC.GateFileError, match='e_do bound'):          # J5-7: no final scoring, no evidence at all
        GC.check(_refresh(d))


def test_ii_iii_missing_evidence_refused_and_liam_drop():
    d = _good()
    _set_ev(d, 'O1-F05', 'LADRC', e_do_xac_dinh_moi_nut=False)
    with pytest.raises(GC.GateFileError, match='C8 is not done'):
        GC.check(_refresh(d))
    d = _good()
    _set_ev(d, 'O1-F05', 'LADRC', g4_can_huu_han_mien_hop_moi_nut=False)
    with pytest.raises(GC.GateFileError, match='CAN LIAM QUYET'):
        GC.check(_refresh(d))
    d['muc_bo_theo_liam'] = {'O1-F05': dict(ngay='2026-11-20', phien='local_x', nguyen_van='FAKE bo muc')}
    GC.check(_refresh(d))
    assert d['levels'] == ['O1-F02', 'O1-F08']


def test_levels_must_match():
    d = _good()
    d['levels'] = ['O1-F02', 'O1-F08']
    with pytest.raises(GC.GateFileError):
        GC.check(d)


def test_phan_loai_du_needed():
    d = _good()
    d['runs_c8']['O1-F02'][0]['chung_cu']['phan_loai_du'] = False
    with pytest.raises(GC.GateFileError, match='phan_loai_du'):
        GC.check(d)


# ── premise of the C8 lock (design section 9), other blocks ───────────────────────────────────────────────────

def test_premise_needs_the_middle_clip():
    d = _good()
    d['tien_de_c8']['clip_h']['giua'] = None
    with pytest.raises(GC.GateFileError, match='CAN LIAM QUYET'):
        GC.check(d)


@pytest.mark.parametrize('edit, msg', [
    (lambda d: d.update(format='gate_c8_v1'), 'refused'),
    (lambda d: d.update(lich_doi_chieu='x'), 'lich_doi_chieu'),
    (lambda d: d['bo_cham']['bang_man_trap'].pop('K'), 'bang_man_trap'),
    (lambda d: d['bo_cham']['bang_man_trap'].update(K=dict(nhanh='alpha')), 'nguon_3b'),
    (lambda d: d['chuan_u']['dieu_kien']['kiem_vi_tri_rieng'].update(trang_thai='thieu'), 'dat'),
    (lambda d: d['chuan_u'].update(le=1.2), 'le'),
    (lambda d: d['camera']['c8b']['moc_dich'].pop('U_moc_tham_psi_rad'), 'moc_dich'),
    (lambda d: d['camera']['c8b']['moc_dich']['nguon']['car_pose'].update(h_m=0.3), 'mien_car_pose'),
    (lambda d: d['camera']['c8b'].pop('eps_H'), 'eps_H'),
    (lambda d: d['ke_hoach_c8'].update(duong_dan='../x.json'), 'inside'),
    (lambda d: d['dieu_2b']['thresholds'].update(g5=5.0), 'expected'),
])
def test_gate_file_refusals(edit, msg):
    d = _good()
    edit(d)
    with pytest.raises(GC.GateFileError, match=msg):
        GC.check(d)


def test_csv_gate_file_refused(tmp_path):
    p = str(tmp_path / 'g.csv')
    with open(p, 'w', encoding='utf-8') as f:
        f.write('a,b\n1,2\n')
    with pytest.raises(GC.GateFileError, match='JSON'):
        GC.read_gate_c8(p)


def test_duplicate_json_keys_refused(tmp_path):
    raw = json.dumps(_good())
    assert '"g5": 7.5' in raw
    p = str(tmp_path / 'dup.json')
    with open(p, 'w', encoding='utf-8') as f:
        f.write(raw.replace('"g5": 7.5', '"g5": 0, "g5": 7.5', 1))
    with pytest.raises(GC.GateFileError, match='duplicate'):
        GC.read_gate_c8(p)


def test_integer_beyond_float():
    d = _good()
    d['runs_c8']['O1-F05'][0]['h_s'] = 10 ** 400
    with pytest.raises(GC.GateFileError, match='float range'):
        GC.check(d)


@pytest.mark.parametrize('level, platform, stage, campaign, want', [
    ('O1-F08', 'SIM', 'script', 'B-chung', o1_gate.CONDS),
    ('O1-F08', 'HW', 'screen', 'A-PID', ('g1', 'g2', 'g3', 'g4')),
    ('O1-F05', 'HW', 'confirm', 'A-PID', o1_gate.CONDS),
    ('O1-K1', 'HW', 'final', 'A-PID', o1_gate.CONDS),
    ('O1-F08', 'HIL', 'hil_check', 'B-PID', ('g1', 'g2', 'g3', 'g4')),
    ('O1-F08', 'HIL', 'final_hil', 'RQ2', o1_gate.CONDS),
])
def test_required_by_run(tmp_path, level, platform, stage, campaign, want):
    gf = GC.read_gate_c8(_write(tmp_path, _good()))
    assert gf.required(level, platform, stage, campaign) == tuple(want)


def test_unknown_platform_refused(tmp_path):
    gf = GC.read_gate_c8(_write(tmp_path, _good()))
    with pytest.raises(GC.GateFileError, match='platform'):
        gf.required('O1-F08', 'CAR', 'final', 'A-PID')


# ── check_nguon on a C8 folder with real source files (design sections 3, 6-10) ──────────────────────────────

def _jw(goc, rel, obj):
    p = os.path.join(goc, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f)
    return dict(duong_dan=rel, sha256=GC.sha256_file(p))


def build_c8(tmp_path, levels=LEVELS, u_t=0.014):
    """A C8 folder (FAKE content, real files) and a gate content that passes check and check_nguon."""
    goc = str(tmp_path / 'C8')
    os.makedirs(goc)
    plan = [dict(o=f'{lv}_{c}', muc=lv, controller=c) for lv in levels for c in GC.CONTROLLERS]
    kh = _jw(goc, 'ke_hoach_c8.json', dict(format=GC.KE_HOACH_FORMAT, o=plan))
    ev = dict(phan_loai_du=True, ban_cham_cuoi=True,
              bo_so=[dict(e_do_lo_vuot_D_moi_nut=False, e_do_xac_dinh_moi_nut=True,
                          g4_can_huu_han_mien_hop_moi_nut=True)])
    ids = ['f1', 'f2', 'w1']
    mien = dict(nut=[0, 1], o_luoi=[0], id_khung=ids, id_khung_sha256=GC.sha256_ids(ids))
    cells, runs, mans = [], {lv: [] for lv in levels}, []
    for x in plan:
        rid = 'R_' + x['o']
        fg = _jw(goc, f'luot/{rid}/gio.json', dict(format=GC.GIO_FORMAT, kiem_gio_d1=dict(trang_thai='dat'),
                                                   dem_truot_kiem_gio=False, h_s=u_t / 2))
        fc = _jw(goc, f'luot/{rid}/luoi.json', dict(trang_thai_luot='cham', outcome='dat', gio_sha256=fg['sha256'],
                                                    u_max_s=u_t / 2, mien_phai_phu=mien))
        cc = _jw(goc, f'luot/{rid}/chung_cu.json', dict(chung_cu_c8=ev, nguon=[dict(file='luoi.json',
                                                                                    sha256=fc['sha256'])]))
        cells.append(dict(o=x['o'], muc=x['muc'], controller=x['controller'], trang_thai='co_luot',
                          chuoi=[dict(run_id=rid, ket_qua='dat', loai_hong=[], kiem_gio_d1='dat',
                                      dem_truot_kiem_gio=False, file_cham=fc, file_gio=fg, file_chung_cu=cc)]))
        runs[x['muc']].append(dict(run_id=rid, controller=x['controller'], o=x['o'], h_s=u_t / 2, u_max_s=u_t / 2,
                                   u_t_s=u_t / 2 + u_t / 2, chung_cu=copy.deepcopy(ev)))
        mans.append(_jw(goc, f'luot/{rid}/u_moc_manifest.json',
                        dict(format=GC.MANIFEST_FORMAT, ok=True, file_cham_sha256=fc['sha256'], mien_da_tinh=mien,
                             U_moc_dong_xy_m=0.004, U_moc_tham_xy_m=0.003, U_moc_dong_psi_rad=0.002,
                             U_moc_tham_psi_rad=0.001)))
    bc = {k: _jw(goc, f'chuan_u/{k}.json', dict(FAKE=k)) for k in GC.CHUAN_U_DK}
    hs = _jw(goc, 'chuan_u/ho_so_c8.json', dict(bang_chung={k: v['sha256'] for k, v in bc.items()},
                                                ke_hoach_c8_sha256=kh['sha256'],
                                                giu_ngoai=[runs[levels[0]][0]['run_id']],
                                                cham_luc={runs[levels[0]][0]['run_id']: '2026-11-21T10:00:00+07:00'},
                                                khoa_kiem_luc='2026-11-21T09:00:00+07:00'))
    clips = [_jw(goc, f'clip_h/H{i}.json', dict(FAKE=i)) for i in range(4)]
    td = _jw(goc, 'tien_de_c8.json', dict(format=GC.TIEN_DE_FORMAT, ket_qua='dat',
                                          sha256=dict(sessions={f'H{i}': c['sha256'] for i, c in enumerate(clips)})))
    d = GC.fake_gate(levels, u_t_s=u_t)
    d.update(c8_dir='C8', written_by=dict(script='c8b_u', commit='FAKE', dirty=False), ke_hoach_c8=kh, o_c8=cells,
             runs_c8=runs, tien_de_c8=dict(clip_h=dict(dau=clips[:2], giua=clips[2], cuoi=clips[3]), file=td))
    d['chuan_u'].update(ho_so_c8=hs, dieu_kien={k: dict(trang_thai='dat', bang_chung=bc[k]) for k in GC.CHUAN_U_DK})
    d['chung_cu_muc'] = {k: dict(e_do={c: [f'R_{k}_{c}'] for c in GC.CONTROLLERS},
                                 g4={c: [f'R_{k}_{c}'] for c in GC.CONTROLLERS}) for k in levels}
    md = d['camera']['c8b']['moc_dich']
    md['nguon']['manifests'] = mans
    md.update(U_moc_dong_xy_m=0.004, U_moc_tham_xy_m=0.003, U_moc_dong_psi_rad=0.002, U_moc_tham_psi_rad=0.001)
    GC.check_nguon(d, goc)
    return d, goc


def test_check_nguon_passes_and_write_gate(tmp_path):
    d, goc = build_c8(tmp_path)
    p = str(tmp_path / 'gate_c8.json')
    sha = GC.write_gate(p, d, goc)
    assert GC.read_gate_c8(p).sha256 == sha == GC.sha256_file(p)
    os.rename(goc, goc + '_moved')                                     # RT4-6: reading needs no source
    assert GC.read_gate_c8(p).levels == list(LEVELS)


def test_write_gate_without_folder_only_for_fake(tmp_path):
    d, goc = build_c8(tmp_path)
    with pytest.raises(GC.GateFileError, match='not FAKE'):
        GC.write_gate(str(tmp_path / 'g.json'), d)
    assert not os.path.exists(str(tmp_path / 'g.json'))
    GC.write_gate(str(tmp_path / 'f.json'), _good())


@pytest.mark.parametrize('target', ['ke_hoach_c8.json', 'luot/R_O1-F05_PID/luoi.json', 'luot/R_O1-F05_PID/gio.json',
                                    'luot/R_O1-F05_PID/chung_cu.json',
                                    'chuan_u/kiem_vi_tri_rieng.json', 'clip_h/H2.json', 'tien_de_c8.json',
                                    'luot/R_O1-F08_LADRC/u_moc_manifest.json'])
def test_rt2_2_a_changed_byte_is_refused_and_not_written(tmp_path, target):
    d, goc = build_c8(tmp_path)
    with open(os.path.join(goc, target), 'ab') as f:
        f.write(b' ')
    with pytest.raises(GC.GateFileError, match='sha256 differs'):
        GC.check_nguon(d, goc)
    with pytest.raises(GC.GateFileError, match='not written'):
        GC.write_gate(str(tmp_path / 'g.json'), d, goc)
    assert not os.path.exists(str(tmp_path / 'g.json'))


def _rewrite(goc, ref, edit):
    p = os.path.join(goc, ref['duong_dan'])
    with open(p, encoding='utf-8') as f:
        x = json.load(f)
    edit(x)
    with open(p, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(x, f)
    ref['sha256'] = GC.sha256_file(p)


def _relink(goc, x):
    """After a time or scoring file was rewritten: tie luoi.json to the time file and chung_cu.json to luoi.json."""
    _rewrite(goc, x['file_cham'], lambda y: y.update(gio_sha256=x['file_gio']['sha256']))
    _rewrite(goc, x['file_chung_cu'], lambda y: y.update(nguon=[dict(file='luoi.json', sha256=x['file_cham']['sha256'])]))


def test_check_nguon_compares_contents(tmp_path):
    d, goc = build_c8(tmp_path / 'a')
    _rewrite(goc, d['o_c8'][2]['chuoi'][0]['file_gio'], lambda x: x['kiem_gio_d1'].update(trang_thai='khong_kiem'))
    _relink(goc, d['o_c8'][2]['chuoi'][0])
    with pytest.raises(GC.GateFileError, match='kiem_gio_d1'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'm2')
    _rewrite(goc, d['o_c8'][2]['chuoi'][0]['file_gio'], lambda x: x.pop('dem_truot_kiem_gio'))
    _relink(goc, d['o_c8'][2]['chuoi'][0])
    with pytest.raises(GC.GateFileError, match='dem_truot_kiem_gio'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'b')
    _rewrite(goc, d['o_c8'][2]['chuoi'][0]['file_gio'], lambda x: x.update(h_s=1.0))
    _relink(goc, d['o_c8'][2]['chuoi'][0])
    with pytest.raises(GC.GateFileError, match='h_s'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'c')
    _rewrite(goc, d['o_c8'][2]['chuoi'][0]['file_chung_cu'],
             lambda x: x['chung_cu_c8']['bo_so'][0].update(g4_can_huu_han_mien_hop_moi_nut=False))
    with pytest.raises(GC.GateFileError, match='chung_cu'):
        GC.check_nguon(d, goc)


def test_check_nguon_f17_scoring_file_ties(tmp_path):
    """F17: luoi.json (outcome, gio_sha256, u_max_s) tied to the run's time file; chung_cu.json made from it."""
    cell = lambda d: d['o_c8'][2]['chuoi'][0]                          # noqa: E731
    d, goc = build_c8(tmp_path / 'a')
    _rewrite(goc, cell(d)['file_cham'], lambda x: x.update(u_max_s=1.0))
    _relink(goc, cell(d))
    with pytest.raises(GC.GateFileError, match='u_max_s'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'b')
    _rewrite(goc, cell(d)['file_cham'], lambda x: x.update(gio_sha256='0' * 64))
    with pytest.raises(GC.GateFileError, match='another time file'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'c')
    _rewrite(goc, cell(d)['file_cham'], lambda x: x.update(outcome='khong_dat'))
    _relink(goc, cell(d))
    with pytest.raises(GC.GateFileError, match='outcome'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'd')
    _rewrite(goc, cell(d)['file_chung_cu'], lambda x: x.update(nguon=[]))
    with pytest.raises(GC.GateFileError, match='made from this'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'e')
    _rewrite(goc, cell(d)['file_cham'], lambda x: x.update(trang_thai_luot='x'))
    with pytest.raises(GC.GateFileError, match='c8b_luoi output'):
        GC.check_nguon(d, goc)


def test_eligible_run_needs_its_evidence_file():
    d = _good()
    d['o_c8'][0]['chuoi'][-1]['file_chung_cu'] = None
    with pytest.raises(GC.GateFileError, match='evidence file'):
        GC.check(d)


def test_ke_hoach_c8_from_the_locked_run_list(tmp_path):
    kh = GC.ke_hoach_c8()
    assert kh['format'] == GC.KE_HOACH_FORMAT and len(kh['o']) == 12
    assert kh['o'][0] == dict(o='C8-P-F02-1', muc='O1-F02', controller='PID')
    assert {(x['muc'], x['controller']) for x in kh['o']} == {(lv, c) for lv in LEVELS for c in GC.CONTROLLERS}
    p = str(tmp_path / 'kh.json')
    assert GC.main(['ke-hoach', '--out', p]) == 0 and json.load(open(p, encoding='utf-8')) == kh
    assert GC.main(['ke-hoach', '--out', p]) == 1                       # never replaced


def test_check_nguon_plan_and_premise(tmp_path):
    d, goc = build_c8(tmp_path / 'a')
    _rewrite(goc, d['ke_hoach_c8'], lambda x: x['o'].append(dict(o='extra', muc='O1-F05', controller='PID')))
    with pytest.raises(GC.GateFileError, match='plan locked'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'b')
    _rewrite(goc, d['tien_de_c8']['file'], lambda x: x.update(ket_qua='khong_dat'))
    with pytest.raises(GC.GateFileError, match='CAN LIAM QUYET'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'c')
    _rewrite(goc, d['tien_de_c8']['file'], lambda x: x['sha256']['sessions'].pop('H2'))
    with pytest.raises(GC.GateFileError, match='four clips'):
        GC.check_nguon(d, goc)


def test_check_nguon_u_standard(tmp_path):
    d, goc = build_c8(tmp_path / 'a')
    _rewrite(goc, d['chuan_u']['ho_so_c8'], lambda x: x['bang_chung'].pop('kiem_vi_tri_rieng'))
    with pytest.raises(GC.GateFileError, match='four evidence'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'b')
    _rewrite(goc, d['chuan_u']['ho_so_c8'], lambda x: x.update(khoa_kiem_luc='2026-11-21T11:00:00+07:00'))
    with pytest.raises(GC.GateFileError, match='not before the first scoring'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'c')
    _rewrite(goc, d['chuan_u']['ho_so_c8'], lambda x: x.update(cham_luc={}))
    with pytest.raises(GC.GateFileError, match='cham_luc'):
        GC.check_nguon(d, goc)


def test_check_nguon_shifted_anchor(tmp_path):
    d, goc = build_c8(tmp_path / 'a')
    m = d['camera']['c8b']['moc_dich']['nguon']['manifests'][0]
    _rewrite(goc, m, lambda x: x['mien_da_tinh']['o_luoi'].append(1))
    with pytest.raises(GC.GateFileError, match='mien_da_tinh'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'b')
    m = d['camera']['c8b']['moc_dich']['nguon']['manifests'][0]
    _rewrite(goc, m, lambda x: x.update(file_cham_sha256='a' * 64))
    with pytest.raises(GC.GateFileError, match='file_cham_sha256'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'c')
    m = d['camera']['c8b']['moc_dich']['nguon']['manifests'][1]
    _rewrite(goc, m, lambda x: x.update(U_moc_tham_psi_rad=0.0011))
    with pytest.raises(GC.GateFileError, match='below the largest'):
        GC.check_nguon(d, goc)


def test_path_outside_the_folder_refused(tmp_path):
    d, goc = build_c8(tmp_path)
    outside = tmp_path / 'outside.json'
    outside.write_text('{}', encoding='utf-8')
    d['ke_hoach_c8'] = dict(duong_dan='sub/../../outside.json', sha256=GC.sha256_file(str(outside)))
    with pytest.raises(GC.GateFileError, match='outside the C8 folder'):
        GC.check_nguon(d, goc)


# ── campaign side with a v2 gate (CODE's choice (a), 19/11) ───────────────────────────────────────────────────

def test_campaign_u_moc_refuses_a_v2_gate(tmp_path):
    p = _write(tmp_path, _good())
    r = dict(run_uid='X', gate_file_sha256=GC.sha256_file(p))
    with pytest.raises(rl.LogError, match='CAMPAIGN_V2_TODO'):
        rl._gate_lock(p, r)


def test_rt4_1_worst_case_slope_needs_the_window_ends(monkeypatch):
    """RT4-1 (Codex round 4, design v5 section 9 item 1): V(s) = L max(s, 0) with L the Lipschitz constant itself,
    delta 1 ms, U_t 27.75 ms: every cell bound covers V at the far end of its window (a grid that stops at the last
    point inside the window, 0.75 delta short, misses it even with the L delta / 2 margin)."""
    sch = Schedule(FIG8['O1-F08'])
    L = 3.0

    def lin(sch_, t, a_m, amp=1.0):
        v = L * np.maximum(np.asarray(t, dtype=float), 0.0)
        z = np.zeros_like(v)
        return v, z, v, z

    monkeypatch.setattr(GC, 'schedule_samples', lin)
    u_t = 0.02775
    blk = _blk(hang=0.0)
    tm = GC._Terms('g3', blk, u_t)
    lip = dict(L_V=L, L_W=0.0, L_u=L, L_r=0.0)
    _, (ta, tb), res = GC._grid_bounds(sch, [tm], lip, blk['a_m'], u_t, -u_t, 0.05, 1e-3)
    assert np.all(res[0][0] >= u_t * L * np.maximum(tb + u_t, 0.0) - 1e-15)
