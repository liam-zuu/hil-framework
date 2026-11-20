"""tools/gate_c8.py: the C8 gate file gate_c8_v2 (check, check_nguon, Uhat, mask, chains, evidence, premise).
CODE_GATE 19/11/2026, design CLAUDE_THESIS 03_CODE/docs/2026-11-19_Thiet_ke_rq3_tuong_doi_v6_C8.md section 12 (after
the scoring rules J v6; tests taken from the counter-examples of Codex, design v5_C8 section 9). Every number is
invented (FAKE)."""
import copy
import hashlib
import json
import math
import os
import shutil
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
    # GC6-10 (round 3): the bounds are rounded outward, so a bound equal to half the threshold within the rounding is
    # not told from it (bien_nguong); just below decides, just above is report only
    uh = GC.uhat_level('O1-F05', _blk('O1-F05', hang=0.05 - 1e-12), 0.01)
    assert uh['g2']['uhat'] > 0.05 - 1e-12 and uh['g2']['trang_thai'] == 'quyet'
    uh = GC.uhat_level('O1-F05', _blk('O1-F05', hang=0.05), 0.01)
    assert uh['g2']['uhat'] > 0.05 > uh['g2']['uhat_duoi'] and uh['g2']['trang_thai'] == 'bien_nguong'
    uh = GC.uhat_level('O1-F05', _blk('O1-F05', hang=0.05 + 1e-12), 0.01)
    assert uh['g2']['trang_thai'] == 'chi_bao_cao'


def test_gc6_10_round3_bound_rounded_out():
    # Codex round 3: V = 1e-17 on a constant 0.05; the exact sum is above the float 0.05, the upper bound must be too
    up = GC._round_out(np.array([0.05 + 1e-17 * 0.027]), up=True)[0]
    assert up > 0.05
    assert GC._round_out(np.array([0.0]), up=False)[0] == 0.0 and GC._round_out(np.array([0.0]), up=True)[0] > 0.0


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
    ref = lambda k: dict(duong_dan=f'FAKE/{rid}_{k}.json', sha256=hashlib.sha256(f'{rid}_{k}'.encode()).hexdigest())  # noqa
    if dl is None:                       # the C8b rule (design v7 7a): a slip counts unless lost by a boot
        dl = d1 == 'truot' and 'thao_tac' not in lh
    return dict(run_id=rid, clip=f'IMG_{rid}', ket_qua=kq, loai_hong=list(lh), kiem_gio_d1=d1, dem_truot_kiem_gio=dl,
                file_cham=ref('luoi'), file_gio=ref('gio') if gio else None,
                file_chung_cu=None if kq == 'hong' else ref('cc'),
                file_so_h=None if kq == 'hong' else dict(duong_dan='FAKE/drift.json', sha256=GC.FAKE_SHA),
                file_cham_h1=None, file_gio_h1=None, file_edo=None if kq == 'hong' else ref('edo'), file_edo_h1=None,
                file_log=ref('log') if gio else None, file_thao_tac=ref('score_tt') if gio else None, thieu_u_t=False)


def _slip(rid, op=False, dl=None):
    return _e(rid, 'hong', ['thao_tac', 'he_do'] if op else ['he_do'], 'truot', dl)


def test_chain_last_lost_run_not_at_cap_refused():
    with pytest.raises(GC.GateFileError, match='run it again'):
        GC.count_chain([_slip('a'), _slip('b')])
    assert GC.count_chain([_slip('a'), _e('b')])[0] == 'co_luot'


def test_gc6_1_a_rerun_replaces_a_lost_run_only():
    """GC6-1 (J v6 section 4 rule 2b): a run that is not lost ends its chain; a valid run never replaces another."""
    with pytest.raises(GC.GateFileError, match='replaces a lost run only'):
        GC.count_chain([_e('a'), _e('b')])
    with pytest.raises(GC.GateFileError, match='replaces a lost run only'):
        GC.count_chain([_e('a', 'khong_dat'), _slip('b'), _e('c')])


def test_gc6_1_files_and_clips_of_one_run_only():
    d = _good()
    d['o_c8'][1]['chuoi'][0]['file_gio'] = copy.deepcopy(d['o_c8'][0]['chuoi'][0]['file_gio'])
    with pytest.raises(GC.GateFileError, match='same file'):
        GC.check(d)
    d = _good()
    d['o_c8'][1]['chuoi'][0]['clip'] = d['o_c8'][0]['chuoi'][0]['clip']
    with pytest.raises(GC.GateFileError, match='one physical run'):
        GC.check(d)


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
    (dict(file_so_h=None), 'file_so_h null'),
    (dict(file_cham_h1=dict(duong_dan='FAKE/x.json', sha256='a' * 64)), 'go together'),
    (dict(clip=''), 'clip'),
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


def _h(s):
    return hashlib.sha256(s.encode()).hexdigest()


def _esp(rid):
    return _h('esp32 ' + rid.split('#')[0])


def _log(goc, rid):
    """The ESP32 log of run rid: a file whose sha256 is _esp(rid) (its bytes are 'esp32 <rid>')."""
    base = rid.split('#')[0]
    p = os.path.join(goc, 'log', base + '.csv')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'wb') as f:
        f.write(('esp32 ' + base).encode())
    return dict(duong_dan=f'log/{base}.csv', sha256=GC.sha256_file(p))


def _score_tt(goc, rid, cell, ops=()):
    """The score.json c8b_gio --score read the operations from (thao_tac, esp32_sha256 of the run's log)."""
    return _jw(goc, f'luot/{rid.replace("#", "_")}/score_tt.json',
               dict(task=cell['muc'], thao_tac=list(ops), esp32_sha256=_esp(rid), outcome='dat'))


def _gio(rid, cell, h_s, tt_ref=None, ops=(), **kw):
    """A c8b_gio_v1 time file of run rid (FAKE numbers; the fields gate_c8 reads)."""
    esp = _esp(rid)
    g = dict(format=GC.GIO_FORMAT, task=cell['muc'], controller=GC.CTRL_LOG[cell['controller']], ok=True, ket_luan='ok',
             kiem_gio_d1=dict(trang_thai='dat', truoc_boot=dict(co_boot='boot' in ops, t_boot_log_s=None,
                                                                  du_cua_so=False, trang_thai=None, thao_tac=list(ops))),
             dem_truot_kiem_gio=False, h_s=h_s, sha256=dict(esp32=esp, frames=_h('frames ' + rid)),
             nguon=[dict(file='frames.csv', sha256=_h('frames ' + rid)), dict(file=rid.split('#')[0] + '.csv', sha256=esp)],
             thao_tac_nguon=None if tt_ref is None else dict(file='score_tt.json', sha256=tt_ref['sha256']),
             nodes_s=[0.0, 0.001], cua_so_ghep_log_s=[1.0, 4.0])
    g.update(kw)
    return g


def _nodes(goc, rid, cell, base=0, fg_obj=None, gio_sha=None, lost=False, ops=(), u_t=None, session=None):
    """Two node scores of a scoring with their mien_phai_phu (CAM F13) and the luoi.json nguon_nut naming them."""
    nn, mpp = [], []
    for k in range(2):
        ids = [base + 10 * k + 1, base + 10 * k + 2]
        m = dict(nut=k, D_s=0.001 * k, id_khung=ids, id_khung_sha256=GC.sha256_ids(ids))
        sc = dict(task=cell['muc'], outcome='hong_he_do' if lost else 'dat', thao_tac=list(ops),
                  esp32_sha256=None if fg_obj is None else fg_obj['sha256']['esp32'],
                  frames_sha256=None if fg_obj is None else fg_obj['sha256']['frames'])
        if not lost:
            sc['khung_tuong_doi_v1'] = dict(mien_phai_phu=m, gio_sha256=gio_sha)
            sc.update(U_t_s=u_t, gate_v3={c: dict(state='dat') for c in o1_gate.CONDS}, session_sha256=session)
        r = _jw(goc, f'luot/{rid}/nut_{k:03d}.json', sc)
        nn.append(dict(nut=k, **r))
        mpp.append(m)
    return nn, mpp


LUC_KHOA_C8B = '2026-11-21T08:00:00+07:00'
LUC_S7 = '2026-11-21T09:00:00+07:00'
LUC_CHAM = '2026-11-21T10:00:00+07:00'


def _luoi(goc, rid, cell, fg, u_max, outcome='dat', clip=None, fg_obj=None, ops=(), session=None, **kw):
    lost = outcome == 'hong_he_do'
    if u_max is None and not lost:
        u_max = 0.005                                   # a scored grid has U_t at every node (GC10-3)
    nn, mpp = _nodes(goc, rid, cell, base=int(_h(rid)[:5], 16) * 100, fg_obj=fg_obj,
                     gio_sha=None if fg is None else fg['sha256'], lost=lost, ops=ops, u_t=u_max, session=session)
    clip = clip or 'IMG_' + rid
    x = dict(trang_thai_luot='hong_he_do' if lost else 'cham', outcome=outcome, task=cell['muc'],
             gio_sha256=None if fg is None else fg['sha256'], u_max_s=u_max, nodes_s=[0.0, 0.001], nguon_nut=nn,
             hong_nut=[lost, lost], clip=dict(ten=clip, clip_sha256=_h('clip ' + clip)),
             script=dict(commit='c' * 40, dirty=False, luc_chay=LUC_CHAM), nodes_ok=[not lost, not lost],
             scorer=None if lost else dict(commit='5' * 40, dirty=False,
                                           nut=[dict(commit='5' * 40, dirty=False, luc_chay=LUC_CHAM)] * 2),
             u_t_nodes=[None, None] if lost else [u_max, u_max], report_only=['g5'] if cell['muc'] == 'O1-F08' else [],
             nums=dict(g4=dict(hi=0.01, n_undef=0)), states={c: 'dat' for c in o1_gate.CONDS},
             file_so_h=None, file_cham_h1=None, file_gio_h1=None,
             loai_hong=(['thao_tac'] if ops else ['he_do']) if lost else [])
    x.update(kw)
    return _jw(goc, f'luot/{rid}/luoi.json', x), mpp


def _write_csv(goc, d):
    """thu_tu_clip.csv from the chains in order (clip, buoc = cell, log_pi = the run's log), plus one row before."""
    lines = ['clip,buoc,gio_pc_bam_quay,gio_pc_dung_quay,created_utc,log_pi,ghi_chu', 'IMG_0001,A2_H0a,,,,,clip H']
    for c in d['o_c8']:
        for j, x in enumerate(c['chuoi']):
            b = c['o'] + (f'_lai{j}' if j else '')                    # a rerun row: <run code>_lai<n> (CODE 19/11)
            lines.append(f'{x["clip"]},{b},,,,{x["run_id"]}.csv,')
    for x in d.get('luot_k') or []:                                  # GC9-2: the K runs (no log for them here)
        lines.append(f'{x["clip"]},{x["run_id"]},,,,,')
    lines.append('IMG_9002,H_m,,,,,clip H giua')
    p = os.path.join(goc, 'thu_tu_clip.csv')
    with open(p, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    d['thu_tu_clip'] = dict(duong_dan='thu_tu_clip.csv', sha256=GC.sha256_file(p))


def _git_commit(goc, rel, when):
    """Commit one file of goc (a git repository made for the test) at the time `when`; returns (commit, %cI time)."""
    import subprocess
    env = dict(os.environ, GIT_AUTHOR_DATE=when, GIT_COMMITTER_DATE=when, GIT_AUTHOR_NAME='t', GIT_AUTHOR_EMAIL='t@t',
               GIT_COMMITTER_NAME='t', GIT_COMMITTER_EMAIL='t@t')
    run = lambda *a: subprocess.run(['git', '-C', goc, *a], check=True, capture_output=True, env=env)  # noqa: E731
    if not os.path.isdir(os.path.join(goc, '.git')):
        run('init', '-q')
        run('config', 'core.autocrlf', 'false')
    run('add', '--', rel)
    run('commit', '-q', '-m', 'khoa_s7', '--', rel)
    c = run('rev-parse', 'HEAD').stdout.decode().strip()
    return c, run('show', '-s', '--format=%cI', c).stdout.decode().strip()


SESSION = dict(width=1920, height=1080, chosen='rational', model_khoa='rational',
               marks={'1': [0.0, 0.0], '2': [1.0, 0.0], '3': [0.0, 1.0]}, fit_ids=[1, 2, 3],
               ref_corners_px={str(i): [[0, 0], [1, 0], [1, 1], [0, 1]] for i in (1, 2, 3)})


def _drift(two, levels, clips, **kw):
    x = dict(kind=GC.DRIFT_KIND, version=GC.DRIFT_VERSION, vuot=two, thieu_H1=False, tasks=list(levels) + ['O1-K1'],
             trang_thai_vi_tri='so', trang_thai_huong='so', D_H_vi_tri_mm=3.0 if two else 0.5, D_H_huong_deg=0.1,
             D_H_mm=3.0 if two else 0.5, eps_H_mm=1.0, eps_H_huong_deg=0.2, vuot_do_thieu=False,
             s0_session_sha256=clips[0]['sha256'], s1_session_sha256=clips[2]['sha256'])
    x.update(kw)
    return x


def build_c8(tmp_path, levels=LEVELS, u_t=0.014, two=False):
    """A C8 folder (FAKE numbers, real files, no FAKE mark) and a gate content that passes check and check_nguon."""
    goc = str(tmp_path / 'C8')
    os.makedirs(goc)
    plan = [dict(o=f'{lv}_{c}', muc=lv, controller=c) for lv in levels for c in GC.CONTROLLERS]
    kh = _jw(goc, 'ke_hoach_c8.json', dict(format=GC.KE_HOACH_FORMAT, o=plan))
    d = GC.fake_gate(levels, u_t_s=u_t)
    names = ('H_a', 'H_b', 'H_m', 'H_z')
    clips = [_jw(goc, f'clip_h/{n}.json', dict(kind=GC.SESSION_KIND, ten=n, video=f'{n}_0{i}_IMG_900{i}.MOV',
                                               video_sha256=_h('video ' + n), **SESSION))
             for i, n in enumerate(names)]
    dr = _jw(goc, 'drift.json', _drift(two, levels, clips))
    ev = dict(phan_loai_du=True, ban_cham_cuoi=True,
              bo_so=[dict(e_do_lo_vuot_D_moi_nut=False, e_do_xac_dinh_moi_nut=True,
                          g4_can_huu_han_mien_hop_moi_nut=True)] * (2 if two else 1))
    edo = lambda g, x: dict(task=x['muc'], trang_thai='dat', vuot_moi_nut=False, can_xac_dinh_moi_nut=True,  # noqa
                            gio_sha256=g['sha256'], esp32_sha256=_esp(rid),
                            theo_nut=[dict(i=k, cham=True, n_undef=0, khung_vuot=[]) for k in range(2)])
    cells, runs, mans = [], {lv: [] for lv in levels}, []
    md = d['camera']['c8b']['moc_dich']
    md['nguon']['car_pose']['phien_ban'] = 'v_test'
    d['camera']['c8b']['mien_car_pose']['phien_ban'] = 'v_test'
    ngm = {k: md['nguon'][k] for k in GC.MOC_NGUON}
    for x in plan:
        rid = 'R_' + x['o']
        tt = _score_tt(goc, rid, x)
        g = _gio(rid, x, u_t / 2, tt)
        fg = _jw(goc, f'luot/{rid}/gio.json', g)
        fg1 = fc1 = fe1 = None
        if two:
            g1 = _gio(rid + '#h1', x, u_t / 2, tt)
            fg1 = _jw(goc, f'luot/{rid}/gio_h1.json', g1)
            fc1, mpp1 = _luoi(goc, rid + '/h1', x, fg1, u_t / 2, clip='IMG_' + rid, fg_obj=g1, session=clips[2]['sha256'])
            fe1 = _jw(goc, f'luot/{rid}/edo_h1.json', edo(fg1, x))
        fc, mpp = _luoi(goc, rid, x, fg, u_t / 2, fg_obj=g, file_so_h=dr, file_cham_h1=fc1, file_gio_h1=fg1)
        fe = _jw(goc, f'luot/{rid}/edo.json', edo(fg, x))
        scorings = [(fc, mpp)] + ([(fc1, mpp1)] if two else [])
        cc = _jw(goc, f'luot/{rid}/chung_cu.json',
                 dict(chung_cu_c8=ev, nguon=[dict(file='luoi.json', sha256=s[0]['sha256']) for s in scorings]
                      + [dict(file='edo.json', sha256=e['sha256']) for e in (fe, fe1) if e is not None]))
        cells.append(dict(o=x['o'], muc=x['muc'], controller=x['controller'], trang_thai='co_luot',
                          chuoi=[dict(run_id=rid, clip='IMG_' + rid, ket_qua='dat', loai_hong=[], kiem_gio_d1='dat',
                                      dem_truot_kiem_gio=False, file_cham=fc, file_gio=fg, file_chung_cu=cc,
                                      file_so_h=dr, file_cham_h1=fc1, file_gio_h1=fg1, file_edo=fe,
                                      file_edo_h1=fe1, file_log=_log(goc, rid), file_thao_tac=tt, thieu_u_t=False)]))
        runs[x['muc']].append(dict(run_id=rid, controller=x['controller'], o=x['o'], h_s=u_t / 2, u_max_s=u_t / 2,
                                   u_t_s=u_t / 2 + u_t / 2, chung_cu=copy.deepcopy(ev)))
        for j, (f, m) in enumerate(scorings):
            mans.append(_jw(goc, f'luot/{rid}/u_moc_manifest_{j}.json',
                            dict(format=GC.MANIFEST_FORMAT, ok=True, file_cham_sha256=f['sha256'], mien_da_tinh=m,
                                 U_moc_dong_xy_m=0.004, U_moc_tham_xy_m=0.003, U_moc_dong_psi_rad=0.002,
                                 U_moc_tham_psi_rad=0.001, nguon=ngm)))
    pl = d['dieu_2b']['per_level']
    d['bo_cham']['phien_ban'] = 'tools/c8b_khoa_v5.py@' + 'c' * 40
    kv5 = _jw(goc, 'c8b_khoa_v5.json', dict(
        version=GC.KHOA_FORMAT, trang_thai='khoa', nhanh_tick=d['u_t_branch'], bang_man_trap=d['bo_cham']['bang_man_trap'],
        muc_kha_thi=list(levels), mien_car_pose=d['camera']['c8b']['mien_car_pose'], eps_H=d['camera']['c8b']['eps_H'],
        khoa_truoc=dict(nguon=[dict(vai='ke_hoach_c8', file='ke_hoach_c8.json', sha256=kh['sha256'])],
                        script=dict(file='tools/c8b_khoa_v5.py', commit='c' * 40, dirty=False), luc=LUC_KHOA_C8B),
        muc={lv: dict(chi_bao_cao=['g5'] if lv == 'O1-F08' else [],
                      hang_dong=dict(a_m=pl[lv]['a_m'], hang_m=pl[lv]['u_p']['hang_m'],
                                     he_so_r=pl[lv]['u_p']['he_so_r'], u_phi_rad=pl[lv]['u_p']['u_phi_rad'],
                                     hang_rad=pl[lv]['u_psi']['hang_rad']),
                      hang_theo_u=[]) for lv in levels}))
    d['camera']['c8b']['khoa_sha256'] = kv5['sha256']
    bc = {k: dict(kv5) for k in GC.DK_KHOA}                             # D1, D4: the C8b lock (C8B 19/11)
    bc['kiem_yaw_theo_hieu_mien_quyet'] = _jw(goc, 'chuan_u/c8b_a6_o.json', dict(version='c8b_a6_o_v1', trang_thai='dat'))
    bc['kiem_vi_tri_rieng'] = _jw(goc, 'chuan_u/c8b_kiem_vi_tri.json', dict(version='c8b_kiem_vi_tri_v1', trang_thai='dat',
                                                                          khoa_truoc_sha256=kv5['sha256']))
    bc = {k: bc[k] for k in GC.CHUAN_U_DK}
    ks = _jw(goc, 'khoa_s7.json', dict(schema='c7_khoa_s7_v1', hang_so_s7={}, commit_hil='d' * 40))
    commit, when = _git_commit(goc, 'khoa_s7.json', LUC_S7)
    kc = _jw(goc, 'kiem_cheo.json', dict(version=GC.KIEM_CHEO_VERSION,
                                         khoa_s7=dict(file='khoa_s7.json', sha256=ks['sha256'], git_commit=commit,
                                                      git_time=when)))
    all_runs = [c['chuoi'][0]['run_id'] for c in cells]
    hs = _jw(goc, 'chuan_u/ho_so_c8.json', dict(format=GC.HO_SO_FORMAT, bang_chung={k: v['sha256'] for k, v in bc.items()},
                                                ke_hoach_c8_sha256=kh['sha256'], khoa_c8b_sha256=kv5['sha256'],
                                                giu_ngoai=all_runs, cham_luc={r: LUC_CHAM for r in all_runs},
                                                khoa_kiem_luc=dict(kiem_cheo=kc, khoa_s7=ks, git_commit=commit,
                                                                   git_time=when)))
    td = _jw(goc, 'tien_de_c8.json', dict(format=GC.TIEN_DE_FORMAT, ket_qua='dat', session_id='C8_buoi2',
                                          vai_tro_clip_h=dict(zip(names, GC.TD_ROLES)),
                                          sha256=dict(sessions={n: c['sha256'] for n, c in zip(names, clips)},
                                                      kiem_cheo=kc['sha256'])))
    d.update(c8_dir='C8', written_by=dict(script='c8b_u', commit='c' * 40, dirty=False), ke_hoach_c8=kh, khoa_c8b=kv5,
             o_c8=cells, runs_c8=runs,
             tien_de_c8=dict(buoi='C8_buoi2', clip_h=dict(dau=clips[:2], giua=clips[2], cuoi=clips[3]), file=td))
    d['chuan_u'].update(ho_so_c8=hs, dieu_kien={k: dict(trang_thai='dat', bang_chung=bc[k]) for k in GC.CHUAN_U_DK})
    d['chung_cu_muc'] = {k: dict(e_do={c: [f'R_{k}_{c}'] for c in GC.CONTROLLERS},
                                 g4={c: [f'R_{k}_{c}'] for c in GC.CONTROLLERS}) for k in levels}
    md['nguon']['manifests'] = mans
    md.update(U_moc_dong_xy_m=0.004, U_moc_tham_xy_m=0.003, U_moc_dong_psi_rad=0.002, U_moc_tham_psi_rad=0.001)
    _write_csv(goc, d)
    GC.check_nguon(d, goc)
    return d, goc


def test_check_nguon_passes_and_write_gate(tmp_path):
    d, goc = build_c8(tmp_path)
    p = str(tmp_path / 'gate_c8.json')
    sha = GC.write_gate(p, d, goc)
    assert GC.read_gate_c8(p).sha256 == sha == GC.sha256_file(p)
    with pytest.raises(GC.GateFileError, match='never replaced'):     # GC6-14
        GC.write_gate(p, d, goc)
    os.rename(goc, goc + '_moved')                                     # RT4-6: reading needs no source
    assert GC.read_gate_c8(p).levels == list(LEVELS)


def test_write_gate_without_folder_only_for_fake(tmp_path):
    d, goc = build_c8(tmp_path)
    with pytest.raises(GC.GateFileError, match='not FAKE'):
        GC.write_gate(str(tmp_path / 'g.json'), d)
    assert not os.path.exists(str(tmp_path / 'g.json'))
    GC.write_gate(str(tmp_path / 'f.json'), _good())
    with pytest.raises(GC.GateFileError, match='FAKE content has no sources'):   # GC6-14
        GC.check_nguon(_good(), goc)
    assert GC.read_gate_c8(str(tmp_path / 'f.json')).fake


@pytest.mark.parametrize('target', ['ke_hoach_c8.json', 'luot/R_O1-F05_PID/luoi.json', 'luot/R_O1-F05_PID/gio.json',
                                    'luot/R_O1-F05_PID/chung_cu.json', 'luot/R_O1-F05_PID/nut_001.json',
                                    'chuan_u/c8b_kiem_vi_tri.json', 'clip_h/H_m.json', 'tien_de_c8.json',
                                    'luot/R_O1-F08_LADRC/u_moc_manifest_0.json', 'thu_tu_clip.csv',
                                    'c8b_khoa_v5.json', 'drift.json'])
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


def _rewrite_khoa(goc, d, edit):
    """Rewrite the C8b lock and the SHA the gate and the dossier carry of it (camera.c8b.khoa_sha256, khoa_c8b_sha256)."""
    _rewrite(goc, d['khoa_c8b'], edit)
    s = d['khoa_c8b']['sha256']
    d['camera']['c8b']['khoa_sha256'] = s
    dk = d['chuan_u']['dieu_kien']
    for k in GC.DK_KHOA:
        dk[k]['bang_chung'] = dict(d['khoa_c8b'])
    _rewrite(goc, dk['kiem_vi_tri_rieng']['bang_chung'], lambda y: y.update(khoa_truoc_sha256=s))
    _rewrite(goc, d['chuan_u']['ho_so_c8'],
             lambda y: y.update(khoa_c8b_sha256=s, bang_chung={k: dk[k]['bang_chung']['sha256'] for k in GC.CHUAN_U_DK}))


def _relink(goc, d, x, old=None, h1=False):
    """After a time or scoring file was rewritten: tie the node scores and luoi.json to the time file, the e_do file to
    the time file, chung_cu.json and the manifest to luoi.json (old: the sha256 of luoi.json the manifest names, when
    luoi.json itself was rewritten; h1: the second scoring and its files)."""
    s = '_h1' if h1 else ''
    fcam, fgio, fedo = x['file_cham' + s], x['file_gio' + s], x['file_edo' + s]
    old = old or fcam['sha256']
    g = None if fgio is None else fgio['sha256']
    go = None if g is None else json.load(open(os.path.join(goc, fgio['duong_dan']), encoding='utf-8'))
    p = os.path.join(goc, fcam['duong_dan'])
    lu = json.load(open(p, encoding='utf-8'))

    def node(y):
        if 'khung_tuong_doi_v1' in y:
            y['khung_tuong_doi_v1']['gio_sha256'] = g
        if go is not None:
            y.update(esp32_sha256=go['sha256']['esp32'], frames_sha256=go['sha256']['frames'])
    for e in lu.get('nguon_nut') or []:
        r = dict(duong_dan=e['duong_dan'], sha256=e['sha256'])
        _rewrite(goc, r, node)
        e['sha256'] = r['sha256']
    _rewrite(goc, fcam, lambda y: y.update(gio_sha256=g, nguon_nut=lu.get('nguon_nut')))
    if fedo is not None and g is not None:
        _rewrite(goc, fedo, lambda y: y.update(gio_sha256=g, esp32_sha256=go['sha256']['esp32']))
    if x['file_chung_cu'] is not None:
        ls = [x['file_cham']] + ([x['file_cham_h1']] if x['file_cham_h1'] else [])
        es = [x['file_edo']] + ([x['file_edo_h1']] if x['file_edo_h1'] else [])
        _rewrite(goc, x['file_chung_cu'], lambda y: y.update(nguon=[dict(file='luoi.json', sha256=r['sha256']) for r in ls]
                                                            + [dict(file='edo.json', sha256=r['sha256']) for r in es]))
    for m in d['camera']['c8b']['moc_dich']['nguon']['manifests']:
        p = os.path.join(goc, m['duong_dan'])
        if json.load(open(p, encoding='utf-8'))['file_cham_sha256'] == old:
            _rewrite(goc, m, lambda y: y.update(file_cham_sha256=fcam['sha256']))


def test_check_nguon_compares_contents(tmp_path):
    d, goc = build_c8(tmp_path / 'a')
    x = d['o_c8'][2]['chuoi'][0]
    _rewrite(goc, x['file_gio'], lambda y: y['kiem_gio_d1'].update(trang_thai='khong_kiem'))
    _relink(goc, d, x)
    with pytest.raises(GC.GateFileError, match='kiem_gio_d1'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'm2')
    x = d['o_c8'][2]['chuoi'][0]
    _rewrite(goc, x['file_gio'], lambda y: y.pop('dem_truot_kiem_gio'))
    _relink(goc, d, x)
    with pytest.raises(GC.GateFileError, match='dem_truot_kiem_gio'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'b')
    x = d['o_c8'][2]['chuoi'][0]
    _rewrite(goc, x['file_gio'], lambda y: y.update(h_s=1.0))
    _relink(goc, d, x)
    with pytest.raises(GC.GateFileError, match='h_s'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'c')
    _rewrite(goc, d['o_c8'][2]['chuoi'][0]['file_chung_cu'],
             lambda y: y['chung_cu_c8']['bo_so'][0].update(g4_can_huu_han_mien_hop_moi_nut=False))
    with pytest.raises(GC.GateFileError, match='chung_cu'):
        GC.check_nguon(d, goc)


def test_check_nguon_f17_scoring_file_ties(tmp_path):
    """F17: luoi.json (outcome, gio_sha256, u_max_s) tied to the run's time file; chung_cu.json made from it."""
    cell = lambda d: d['o_c8'][2]['chuoi'][0]                          # noqa: E731
    d, goc = build_c8(tmp_path / 'a')
    _rewrite(goc, cell(d)['file_cham'], lambda x: x.update(u_max_s=1.0))
    _relink(goc, d, cell(d))
    with pytest.raises(GC.GateFileError, match='u_max_s'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'b')
    _rewrite(goc, cell(d)['file_cham'], lambda x: x.update(gio_sha256='0' * 64))
    with pytest.raises(GC.GateFileError, match='another time file'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'b2')                                 # GC6-15: absent is not null
    _rewrite(goc, cell(d)['file_cham'], lambda x: x.pop('gio_sha256'))
    with pytest.raises(GC.GateFileError, match='no gio_sha256'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'c')
    _rewrite(goc, cell(d)['file_cham'], lambda x: x.update(outcome='khong_dat'))
    _relink(goc, d, cell(d))
    with pytest.raises(GC.GateFileError, match='outcome'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'd')
    _rewrite(goc, cell(d)['file_chung_cu'], lambda x: x.update(nguon=[]))
    with pytest.raises(GC.GateFileError, match='made from'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'e')
    _rewrite(goc, cell(d)['file_cham'], lambda x: x.update(trang_thai_luot='x'))
    with pytest.raises(GC.GateFileError, match='c8b_luoi output'):
        GC.check_nguon(d, goc)


def test_eligible_run_needs_its_evidence_file():
    d = _good()
    d['o_c8'][0]['chuoi'][-1].update(file_chung_cu=None, file_edo=None)
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
    _rewrite(goc, d['tien_de_c8']['file'], lambda x: x['sha256']['sessions'].pop('H_m'))
    with pytest.raises(GC.GateFileError, match='four roles'):
        GC.check_nguon(d, goc)


def test_gc6_4_premise_by_role_and_session(tmp_path):
    """GC6-4: the four clips H compared role by role (dau1, dau2, giua, cuoi), of this session, readable."""
    d, goc = build_c8(tmp_path / 'a')
    ch = d['tien_de_c8']['clip_h']
    ch['giua'], ch['cuoi'] = ch['cuoi'], ch['giua']                    # same bytes, roles swapped
    with pytest.raises(GC.GateFileError, match='role giua|in the middle|part of the session'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'b')
    d['tien_de_c8']['buoi'] = 'C8_buoi1'
    with pytest.raises(GC.GateFileError, match='premise of session'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'c')
    _rewrite(goc, d['tien_de_c8']['clip_h']['dau'][1], lambda x: x.pop('fit_ids'))
    _rewrite(goc, d['tien_de_c8']['file'],
             lambda x: x['sha256']['sessions'].update(H_b=d['tien_de_c8']['clip_h']['dau'][1]['sha256']))
    with pytest.raises(GC.GateFileError, match='camera session file'):
        GC.check_nguon(d, goc)


def test_check_nguon_u_standard(tmp_path):
    d, goc = build_c8(tmp_path / 'a')
    _rewrite(goc, d['chuan_u']['ho_so_c8'], lambda x: x['bang_chung'].pop('kiem_vi_tri_rieng'))
    with pytest.raises(GC.GateFileError, match='four evidence'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'c')
    _rewrite(goc, d['chuan_u']['ho_so_c8'], lambda x: x.update(cham_luc={}))
    with pytest.raises(GC.GateFileError, match='cham_luc'):
        GC.check_nguon(d, goc)


def _s7_commit(goc, d, when, stated=None):
    """Commit the S7 manifest again (an empty commit, same bytes) at `when`; kiem_cheo.json and the dossier name it,
    with the time `stated` (default: the true commit time)."""
    import subprocess
    env = dict(os.environ, GIT_AUTHOR_DATE=when, GIT_COMMITTER_DATE=when, GIT_AUTHOR_NAME='t', GIT_AUTHOR_EMAIL='t@t',
               GIT_COMMITTER_NAME='t', GIT_COMMITTER_EMAIL='t@t')
    subprocess.run(['git', '-C', goc, 'commit', '-q', '--allow-empty', '-m', 'late'], check=True, env=env)
    c = subprocess.run(['git', '-C', goc, 'rev-parse', 'HEAD'], check=True, capture_output=True).stdout.decode().strip()
    t = stated or subprocess.run(['git', '-C', goc, 'show', '-s', '--format=%cI', c], check=True,
                                 capture_output=True).stdout.decode().strip()
    kk = json.load(open(os.path.join(goc, d['chuan_u']['ho_so_c8']['duong_dan']), encoding='utf-8'))['khoa_kiem_luc']
    kc = dict(kk['kiem_cheo'])
    _rewrite(goc, kc, lambda y: y['khoa_s7'].update(git_commit=c, git_time=t))
    _rewrite(goc, d['chuan_u']['ho_so_c8'], lambda y: y['khoa_kiem_luc'].update(kiem_cheo=kc, git_commit=c, git_time=t))


def test_round3_s7_lock_read_back_from_git(tmp_path):
    """GC6-1 (round 3), KCJ1-14: the S7 lock time is the commit time read from git, before the first held-out scoring."""
    d, goc = build_c8(tmp_path / 'a')
    _s7_commit(goc, d, '2026-11-21T11:00:00+07:00')                    # committed after the scoring
    with pytest.raises(GC.GateFileError, match='not before the first scoring'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'b')
    _s7_commit(goc, d, '2026-11-21T11:00:00+07:00', stated='2026-11-21T09:00:00+07:00')   # an earlier time stated
    with pytest.raises(GC.GateFileError, match='not the time of its commit'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'c')                                  # the manifest changed after its commit
    kk = json.load(open(os.path.join(goc, d['chuan_u']['ho_so_c8']['duong_dan']), encoding='utf-8'))['khoa_kiem_luc']
    ks, kc = dict(kk['khoa_s7']), dict(kk['kiem_cheo'])
    _rewrite(goc, ks, lambda y: y.update(hang_so_s7={'n': 1}))
    _rewrite(goc, kc, lambda y: y['khoa_s7'].update(sha256=ks['sha256']))
    _rewrite(goc, d['chuan_u']['ho_so_c8'], lambda y: y['khoa_kiem_luc'].update(kiem_cheo=kc, khoa_s7=ks))
    with pytest.raises(GC.GateFileError, match='committed unchanged'):
        GC.check_nguon(d, goc)


def test_u_standard_evidence_files_c8b_signature(tmp_path):
    """C8B 19/11: D1, D4 the C8b lock; D2 c8b_a6_o_v1, D3 c8b_kiem_vi_tri_v1 saying dat; D4 one clean scorer commit."""
    dk = lambda d: d['chuan_u']['dieu_kien']                             # noqa: E731
    d, goc = build_c8(tmp_path / 'a')
    dk(d)['khoa_kiem_truoc_giu_ngoai']['bang_chung'] = dict(dk(d)['kiem_vi_tri_rieng']['bang_chung'])
    _rewrite(goc, d['chuan_u']['ho_so_c8'], lambda y: y['bang_chung'].update(
        khoa_kiem_truoc_giu_ngoai=dk(d)['kiem_vi_tri_rieng']['bang_chung']['sha256']))
    with pytest.raises(GC.GateFileError, match='the C8b lock'):
        GC.check_nguon(d, goc)
    for name, k, edit, msg in [('b', 'kiem_yaw_theo_hieu_mien_quyet', lambda y: y.update(trang_thai='thieu'), 'dat'),
                               ('c', 'kiem_vi_tri_rieng', lambda y: y.update(version='x'), 'c8b_kiem_vi_tri_v1'),
                               ('e', 'kiem_vi_tri_rieng', lambda y: y.update(khoa_truoc_sha256='a' * 64), 'khoa_truoc')]:
        d, goc = build_c8(tmp_path / name)
        r = dk(d)[k]['bang_chung']
        _rewrite(goc, r, edit)
        _rewrite(goc, d['chuan_u']['ho_so_c8'], lambda y: y['bang_chung'].update({k: r['sha256']}))
        with pytest.raises(GC.GateFileError, match=msg):
            GC.check_nguon(d, goc)
    for name, sc, msg in [('f', dict(commit='5' * 40, dirty=True, nut=[]), 'scorer'),
                          ('g', dict(commit='6' * 40, dirty=False, nut=[]), 'one clean commit')]:
        d, goc = build_c8(tmp_path / name)
        x = d['o_c8'][0]['chuoi'][0]
        old = x['file_cham']['sha256']
        _rewrite(goc, x['file_cham'], lambda y: y.update(scorer=sc))
        _relink(goc, d, x, old)
        with pytest.raises(GC.GateFileError, match=msg):
            GC.check_nguon(d, goc)


def test_round3_c8b_lock_before_the_first_scoring(tmp_path):
    """GC6-1 (round 3): khoa_truoc.luc of the C8b lock after the first scoring is refused."""
    d, goc = build_c8(tmp_path)
    _rewrite_khoa(goc, d, lambda k: k['khoa_truoc'].update(luc='2026-11-21T10:30:00+07:00'))
    with pytest.raises(GC.GateFileError, match='not before the first scoring of a C8 run'):
        GC.check_nguon(d, goc)


def test_check_nguon_shifted_anchor(tmp_path):
    d, goc = build_c8(tmp_path / 'a')
    m = d['camera']['c8b']['moc_dich']['nguon']['manifests'][0]
    _rewrite(goc, m, lambda x: x['mien_da_tinh'][0]['id_khung'].append(99))
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


def test_gc6_6_every_scoring_has_one_manifest(tmp_path):
    """GC6-6: the lock is the maximum over every scoring of every eligible run: the manifest holding the maximum
    cannot be left out, none twice, and the covered domain is read from the node scores (nguon_nut)."""
    d, goc = build_c8(tmp_path / 'a')
    mans = d['camera']['c8b']['moc_dich']['nguon']['manifests']
    mans.pop()
    with pytest.raises(GC.GateFileError, match='no U manifest'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'b')
    mans = d['camera']['c8b']['moc_dich']['nguon']['manifests']
    mans.append(_jw(goc, 'dup_manifest.json', json.load(open(os.path.join(goc, mans[0]['duong_dan']), encoding='utf-8'))))
    with pytest.raises(GC.GateFileError, match='second manifest'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'c')
    x = d['o_c8'][0]['chuoi'][0]
    old = x['file_cham']['sha256']
    _rewrite(goc, x['file_cham'], lambda y: y.pop('nguon_nut'))
    _relink(goc, d, x, old)
    with pytest.raises(GC.GateFileError, match='nguon_nut'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'd')
    m = d['camera']['c8b']['moc_dich']['nguon']['manifests'][0]
    _rewrite(goc, m, lambda y: y.update(mien_da_tinh=[]))
    with pytest.raises(GC.GateFileError, match='mien_da_tinh'):
        GC.check_nguon(d, goc)


def test_path_outside_the_folder_refused(tmp_path):
    d, goc = build_c8(tmp_path)
    outside = tmp_path / 'outside.json'
    outside.write_text('{}', encoding='utf-8')
    d['ke_hoach_c8'] = dict(duong_dan='sub/../../outside.json', sha256=GC.sha256_file(str(outside)))
    with pytest.raises(GC.GateFileError, match='inside it'):
        GC.check_nguon(d, goc)


@pytest.mark.parametrize('p', ['FAKE/../../x.json', 'a/./b.json', 'C:/x.json', 'C:x.json', '//srv/x.json', '/x.json',
                               'a//b.json'])
def test_gc6_14_reference_paths(p):
    with pytest.raises(GC.GateFileError, match='inside it'):
        GC._ref(dict(duong_dan=p, sha256='a' * 64), 'x')


# ── GC6-1, GC6-2: the physical run order and the lost kinds from the sources ──────────────────────────────────

def _add_lost(goc, d, ci, tag, lh, luoi_kw=None, gio_kw=None, gio=True, d1='dat', dem=False, ops=()):
    """Put a lost run before the run of cell ci (its own time file, log, score.json of its operations and luoi.json),
    rewrite thu_tu_clip.csv."""
    c = d['o_c8'][ci]
    rid = f'L{tag}_{c["o"]}'
    tt = _score_tt(goc, rid, c, ops) if gio else None
    go = _gio(rid, c, 0.005, tt, ops=ops, **(gio_kw or {})) if gio else None
    fg = _jw(goc, f'luot/{rid}/gio.json', go) if gio else None
    kw = dict(luoi_kw or {})
    oc = kw.pop('outcome', 'hong_he_do')
    fc, _ = _luoi(goc, rid, c, fg, None, outcome=oc, fg_obj=go, ops=ops, **kw)
    c['chuoi'].insert(0, dict(run_id=rid, clip='IMG_' + rid, ket_qua='hong', loai_hong=list(lh), kiem_gio_d1=d1,
                              dem_truot_kiem_gio=dem, file_cham=fc, file_gio=fg, file_chung_cu=None, file_so_h=None,
                              file_cham_h1=None, file_gio_h1=None, file_edo=None, file_edo_h1=None,
                              file_log=_log(goc, rid), file_thao_tac=tt, thieu_u_t=False))
    _write_csv(goc, d)
    _fix_ho_so(goc, d)
    return c['chuoi'][0]


def _fix_ho_so(goc, d):
    """The dossier names every C8 run (cham_luc, giu_ngoai): after a run was added."""
    runs = [x['run_id'] for c in d['o_c8'] for x in c['chuoi']]
    _rewrite(goc, d['chuan_u']['ho_so_c8'], lambda y: y.update(giu_ngoai=runs, cham_luc={r: LUC_CHAM for r in runs}))


def _csv_edit(goc, d, edit):
    p = os.path.join(goc, 'thu_tu_clip.csv')
    lines = open(p, encoding='utf-8').read().splitlines()
    lines = edit(lines)
    with open(p, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    d['thu_tu_clip']['sha256'] = GC.sha256_file(p)


def test_gc6_1_chain_against_the_clip_log(tmp_path):
    d, goc = build_c8(tmp_path / 'a')
    _add_lost(goc, d, 0, 1, ['he_do'], luoi_kw=dict(loai_hong=['he_do']))
    GC.check_nguon(d, goc)                                             # a lost run then its rerun: passes
    _csv_edit(goc, d, lambda ls: ls[:2] + [ls[3], ls[2]] + ls[4:])      # rerun logged before the lost run
    with pytest.raises(GC.GateFileError, match='in that order|GC9-4'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'b')                                  # a physical run left out of the chain
    _csv_edit(goc, d, lambda ls: ls[:2] + [f'IMG_X,{d["o_c8"][0]["o"]},,,,X.csv,'] + ls[2:])
    with pytest.raises(GC.GateFileError, match='none left out|GC9-4'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'c')                                  # a figure-8 run outside the plan
    _csv_edit(goc, d, lambda ls: ls + ['IMG_Y,C8-P-F05-3,,,,Y.csv,'])
    with pytest.raises(GC.GateFileError, match='no cell of the plan'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'd')                                  # the time file reads another log
    _csv_edit(goc, d, lambda ls: [ls[0], ls[1], ls[2].replace('.csv,', '_other.csv,')] + ls[3:])
    with pytest.raises(GC.GateFileError, match='ESP32 log|thu_tu_clip.csv names'):
        GC.check_nguon(d, goc)


def test_gc6_1_sources_of_another_task_or_controller(tmp_path):
    """Counter-example of Codex: the whole luoi / gio / chung_cu bundle of an F08 run given for an F05 cell."""
    d, goc = build_c8(tmp_path / 'a')
    f05 = next(c for c in d['o_c8'] if c['o'] == 'O1-F05_PID')
    _rewrite(goc, f05['chuoi'][0]['file_gio'], lambda y: y.update(task='O1-F08'))
    _relink(goc, d, f05['chuoi'][0])
    with pytest.raises(GC.GateFileError, match='task'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'b')
    f05 = next(c for c in d['o_c8'] if c['o'] == 'O1-F05_PID')
    _rewrite(goc, f05['chuoi'][0]['file_cham'], lambda y: y.update(task='O1-F08'))
    _relink(goc, d, f05['chuoi'][0])
    with pytest.raises(GC.GateFileError, match='task'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'c')
    f05 = next(c for c in d['o_c8'] if c['o'] == 'O1-F05_PID')
    _rewrite(goc, f05['chuoi'][0]['file_gio'], lambda y: y.update(controller='ADRC'))
    _relink(goc, d, f05['chuoi'][0])
    with pytest.raises(GC.GateFileError, match='controller'):
        GC.check_nguon(d, goc)


def test_gc6_5_the_c8b_lock(tmp_path):
    """GC6-5: closed terms, shutter table, tick branch copied from the C8b lock; the plan anchored in khoa_truoc."""
    for name, edit, msg in [
            ('a', lambda k: k['muc']['O1-F05']['hang_dong'].update(hang_rad=0.001), 'hang_dong of the C8b lock'),
            ('b', lambda k: k['muc']['O1-F05'].pop('hang_dong'), 'missing'),
            ('b2', lambda k: k['muc']['O1-F05'].update(hang_theo_u=[dict(u_lo_m_s=0.0, u_hi_m_s=9.0, P4_m=0.001,
                                                                        A4_deg=0.1, trang_thai='dat')]), 'hang_theo_u'),
            ('c', lambda k: k.update(trang_thai='chua_khoa'), 'not locked'),
            ('d', lambda k: k['bang_man_trap'].update(K=dict(nhanh='alpha')), 'bang_man_trap'),
            ('e', lambda k: k.update(nhanh_tick='giu_tick'), 'u_t_branch'),
            ('f', lambda k: k['khoa_truoc'].update(nguon=[]), 'anchored'),
            ('f2', lambda k: k['khoa_truoc'].update(nguon=[dict(vai='3b O1-F05', file='x.json', sha256='a' * 64)]),
             'anchored'),
            ('f3', lambda k: k['khoa_truoc']['nguon'][0].update(vai='3b O1-F05'), 'anchored'),   # the plan SHA, other role
            ('g', lambda k: k['muc']['O1-F08'].update(chi_bao_cao=[]), 'F08')]:
        d, goc = build_c8(tmp_path / name)
        _rewrite_khoa(goc, d, edit)
        with pytest.raises(GC.GateFileError, match=msg):
            GC.check_nguon(d, goc)


def test_gc6_2_lost_kinds_from_the_sources(tmp_path):
    # GC9-5 (round 4): luoi.json loai_hong recomputed from its node scores (c8b_luoi rule): none named on a grid lost at
    # every node, or another kind than the node scores give, is refused
    for name, lh in (('a', []), ('b', ['cua_so_tham_chieu'])):
        d, goc = build_c8(tmp_path / name)
        _add_lost(goc, d, 0, 1, ['he_do'], luoi_kw=dict(loai_hong=lh))
        with pytest.raises(GC.GateFileError, match='c8b_luoi rule'):
            GC.check_nguon(d, goc)
    # a run the sources show valid, declared lost (removed from the evidence)
    d, goc = build_c8(tmp_path / 'c')
    _add_lost(goc, d, 0, 1, ['cua_so_tham_chieu'], luoi_kw=dict(outcome='dat'))
    with pytest.raises(GC.GateFileError, match='sources show'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'c2')                                 # Codex round 2: kinds on a scored grid
    _add_lost(goc, d, 0, 1, ['cua_so_tham_chieu'], luoi_kw=dict(outcome='dat', loai_hong=['cua_so_tham_chieu']))
    with pytest.raises(GC.GateFileError, match='c8b_luoi rule'):
        GC.check_nguon(d, goc)
    # an operation the time file states that score.json does not (score.json wins, CODE 19/11)
    d, goc = build_c8(tmp_path / 'd')
    x = d['o_c8'][0]['chuoi'][0]
    _rewrite(goc, x['file_gio'], lambda y: y['kiem_gio_d1']['truoc_boot'].update(thao_tac=['loi_nap_gain']))
    _relink(goc, d, x)
    with pytest.raises(GC.GateFileError, match='score.json wins'):
        GC.check_nguon(d, goc)
    # an operation of score.json on a run the grid scored and the gate keeps
    d, goc = build_c8(tmp_path / 'd2')
    x = d['o_c8'][0]['chuoi'][0]
    _ops(goc, d, x, ['loi_nap_gain'])
    with pytest.raises(GC.GateFileError, match='grid scored the run'):
        GC.check_nguon(d, goc)
    # Codex round 3: the grid names thao_tac, score.json and the time file none
    d, goc = build_c8(tmp_path / 'd3')
    _add_lost(goc, d, 0, 1, ['thao_tac', 'he_do'], luoi_kw=dict(loai_hong=['thao_tac', 'he_do']))
    with pytest.raises(GC.GateFileError, match='c8b_luoi rule'):
        GC.check_nguon(d, goc)
    # a node score with other operations than score.json
    d, goc = build_c8(tmp_path / 'd4')
    x = _add_lost(goc, d, 0, 1, ['thao_tac'], ops=('boot',))
    GC.check_nguon(d, goc)
    lu = json.load(open(os.path.join(goc, x['file_cham']['duong_dan']), encoding='utf-8'))
    r = dict(duong_dan=lu['nguon_nut'][0]['duong_dan'], sha256=lu['nguon_nut'][0]['sha256'])
    _rewrite(goc, r, lambda y: y.update(thao_tac=[]))
    lu['nguon_nut'][0]['sha256'] = r['sha256']
    _rewrite(goc, x['file_cham'], lambda y: y.update(nguon_nut=lu['nguon_nut']))
    with pytest.raises(GC.GateFileError, match='node score states other operations|c8b_luoi rule'):
        GC.check_nguon(d, goc)
    # the time file lost (ok false, hong_he_do): he_do for the run
    d, goc = build_c8(tmp_path / 'e2')
    _add_lost(goc, d, 0, 1, ['he_do'], gio_kw=dict(ok=False, ket_luan='hong_he_do'))
    GC.check_nguon(d, goc)
    # GC9-6 (round 4): a scored grid with a lost node carries that node's kind (c8b_luoi), the run is not lost
    fc = dict(trang_thai_luot='cham', loai_hong=['he_do'])
    assert GC._lost_kinds(fc, None, [], [dict(outcome='dat'), dict(outcome='hong_he_do')], 'x') == set()
    with pytest.raises(GC.GateFileError, match='c8b_luoi rule'):
        GC._lost_kinds(dict(fc, loai_hong=[]), None, [], [dict(outcome='dat'), dict(outcome='hong_he_do')], 'x')
    # M2 counted on a gain load error without a boot (Codex round 2)
    lk = dict(loai_hong=['thao_tac'])
    for name, ops, tb, dem, msg in [
            ('e3', ('loi_nap_gain',), dict(co_boot=False, t_boot_log_s=None, du_cua_so=True, trang_thai='truot',
                                           thao_tac=['loi_nap_gain']), True, 'M2|GC6-2'),
            # a slip counted on a boot run needs the locked check run before the boot over its whole window
            ('f', ('boot',), dict(co_boot=True, t_boot_log_s=5.0, du_cua_so=False, trang_thai='khong_kiem',
                                  thao_tac=['boot']), True, 'M2'),
            # Codex round 3: co_boot false on a boot run, the slip counted
            ('f2', ('boot',), dict(co_boot=False, t_boot_log_s=None, du_cua_so=False, trang_thai=None,
                                   thao_tac=['boot']), True, 'co_boot'),
            # an independent slip before the boot not counted: refused too (the counter is recomputed both ways)
            ('f3', ('boot',), dict(co_boot=True, t_boot_log_s=5.0, du_cua_so=True, trang_thai='truot',
                                   thao_tac=['boot']), False, 'M2'),
            # Codex round 4: a boot without its time in the log, the whole window claimed before it
            ('f4', ('boot',), dict(co_boot=True, t_boot_log_s=None, du_cua_so=True, trang_thai='truot',
                                   thao_tac=['boot']), True, 'without a boot time')]:
        d, goc = build_c8(tmp_path / name)
        _add_lost(goc, d, 0, 1, ['thao_tac', 'he_do'], d1='truot', dem=dem, ops=ops, luoi_kw=lk,
                  gio_kw=dict(ok=False, ket_luan='hong_he_do', dem_truot_kiem_gio=dem,
                              kiem_gio_d1=dict(trang_thai='truot', truoc_boot=tb)))
        with pytest.raises(GC.GateFileError, match=msg):
            GC.check_nguon(d, goc)
    # M2 with the independent evidence: counted, accepted
    d, goc = build_c8(tmp_path / 'g')
    tb = dict(co_boot=True, t_boot_log_s=5.0, du_cua_so=True, trang_thai='truot', thao_tac=['boot'])
    _add_lost(goc, d, 0, 1, ['thao_tac', 'he_do'], d1='truot', dem=True, ops=('boot',), luoi_kw=lk,
              gio_kw=dict(ok=False, ket_luan='hong_he_do', dem_truot_kiem_gio=True,
                          kiem_gio_d1=dict(trang_thai='truot', truoc_boot=tb)))
    GC.check_nguon(d, goc)


def _ops(goc, d, x, ops):
    """The operations of run x set in its score.json, time file and node scores (the sources agree), relinked."""
    _rewrite(goc, x['file_thao_tac'], lambda y: y.update(thao_tac=list(ops)))
    _rewrite(goc, x['file_gio'], lambda y: (y['kiem_gio_d1']['truoc_boot'].update(thao_tac=list(ops),
                                                                                   co_boot='boot' in ops),
                                            y['thao_tac_nguon'].update(sha256=x['file_thao_tac']['sha256'])))
    lu = json.load(open(os.path.join(goc, x['file_cham']['duong_dan']), encoding='utf-8'))
    for e in lu['nguon_nut']:
        r = dict(duong_dan=e['duong_dan'], sha256=e['sha256'])
        _rewrite(goc, r, lambda y: y.update(thao_tac=list(ops)))
        e['sha256'] = r['sha256']
    old = x['file_cham']['sha256']
    _rewrite(goc, x['file_cham'], lambda y: y.update(nguon_nut=lu['nguon_nut'], loai_hong=['thao_tac'] if ops else []))
    _relink(goc, d, x, old)


# ── GC6-3: the clip H comparison and the final scoring ───────────────────────────────────────────────────────

def test_gc6_3_two_scorings(tmp_path):
    d, goc = build_c8(tmp_path / 'a', two=True)                        # above eps_H: two scorings, two sets
    p = str(tmp_path / 'g.json')
    GC.write_gate(p, d, goc)
    # H0 dat, H1 khong_dat: the final outcome is chua_ket_luan (calib_drift_split), not dat
    d, goc = build_c8(tmp_path / 'b', two=True)
    x = d['o_c8'][0]['chuoi'][0]
    lu = json.load(open(os.path.join(goc, x['file_cham_h1']['duong_dan']), encoding='utf-8'))
    for e in lu['nguon_nut']:                                       # the H1 nodes judge G2 khong_dat
        r = dict(duong_dan=e['duong_dan'], sha256=e['sha256'])
        _rewrite(goc, r, lambda y: y['gate_v3']['g2'].update(state='khong_dat'))
        e['sha256'] = r['sha256']
    old1 = x['file_cham_h1']['sha256']
    _rewrite(goc, x['file_cham_h1'], lambda y: (y.update(outcome='khong_dat', nguon_nut=lu['nguon_nut']),
                                                y['states'].update(g2='khong_dat')))
    _relink(goc, d, x, old1, h1=True)
    _rewrite(goc, x['file_chung_cu'], lambda y: y.update(nguon=[dict(file='luoi.json', sha256=x['file_cham']['sha256']),
                                                               dict(file='luoi.json', sha256=x['file_cham_h1']['sha256'])]))
    with pytest.raises(GC.GateFileError, match="outcome is 'chua_ket_luan'"):
        GC.check_nguon(d, goc)
    # two scorings giving two U_t: J does not decide which
    d, goc = build_c8(tmp_path / 'c', two=True)
    x = d['o_c8'][0]['chuoi'][0]
    _rewrite(goc, x['file_gio_h1'], lambda y: y.update(h_s=0.009))
    _relink(goc, d, x, h1=True)
    with pytest.raises(GC.GateFileError, match='CAN LIAM QUYET'):
        GC.check_nguon(d, goc)
    # the comparison says two scorings, the chain gives one
    d, goc = build_c8(tmp_path / 'd')
    _rewrite(goc, d['o_c8'][0]['chuoi'][0]['file_so_h'], lambda y: y.update(vuot=True, D_H_vi_tri_mm=3.0))
    for c in d['o_c8']:
        c['chuoi'][0]['file_so_h']['sha256'] = d['o_c8'][0]['chuoi'][0]['file_so_h']['sha256']
    with pytest.raises(GC.GateFileError, match='two scorings'):
        GC.check_nguon(d, goc)


def test_gc6_3_clip_h_at_the_end_missing(tmp_path):
    d, goc = build_c8(tmp_path / 'a')
    _rewrite(goc, d['o_c8'][0]['chuoi'][0]['file_so_h'], lambda y: y.update(vuot=True, thieu_H1=True, D_H_mm=None))
    for c in d['o_c8']:
        c['chuoi'][0]['file_so_h']['sha256'] = d['o_c8'][0]['chuoi'][0]['file_so_h']['sha256']
    with pytest.raises(GC.GateFileError, match="outcome is 'chua_ket_luan'"):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'b')
    _rewrite(goc, d['o_c8'][0]['chuoi'][0]['file_so_h'], lambda y: y.update(tasks=['O1-F02']))
    for c in d['o_c8']:
        c['chuoi'][0]['file_so_h']['sha256'] = d['o_c8'][0]['chuoi'][0]['file_so_h']['sha256']
    with pytest.raises(GC.GateFileError, match='covers'):
        GC.check_nguon(d, goc)


def test_gc6_3_evidence_made_from_the_scorings_of_its_run_only(tmp_path):
    d, goc = build_c8(tmp_path)
    x, y = d['o_c8'][0]['chuoi'][0], d['o_c8'][1]['chuoi'][0]
    _rewrite(goc, x['file_chung_cu'], lambda z: z['nguon'].append(dict(file='luoi.json', sha256=y['file_cham']['sha256'])))
    with pytest.raises(GC.GateFileError, match='made from'):
        GC.check_nguon(d, goc)


def test_gc6_3_ban_cham_cuoi_and_sets_from_the_comparison(tmp_path):
    d, goc = build_c8(tmp_path / 'a')
    x = d['o_c8'][0]['chuoi'][0]
    r = next(r for r in d['runs_c8'][d['o_c8'][0]['muc']] if r['run_id'] == x['run_id'])
    r['chung_cu']['bo_so'] = r['chung_cu']['bo_so'] * 2
    _rewrite(goc, x['file_chung_cu'], lambda y: y.update(chung_cu_c8=r['chung_cu']))
    with pytest.raises(GC.GateFileError, match='recomputed from the scorings'):
        GC.check_nguon(d, goc)


@pytest.mark.parametrize('edit, field', [
    (lambda e: (e.update(vuot_moi_nut=True), [n.update(khung_vuot=[1]) for n in e['theo_nut']]),
     'e_do_lo_vuot_D_moi_nut'),                                               # Codex round 3: the level would go
    (lambda e: e.update(can_xac_dinh_moi_nut=False), 'e_do_xac_dinh_moi_nut'),
    (lambda e: e.update(trang_thai='chua_ket_luan'), 'e_do_xac_dinh_moi_nut')])
def test_round3_evidence_recomputed_from_the_edo_files(tmp_path, edit, field):
    """GC6-3 (round 3): the evidence is recomputed from the e_do file of each scoring, never copied alone."""
    d, goc = build_c8(tmp_path)
    x = d['o_c8'][0]['chuoi'][0]
    _rewrite(goc, x['file_edo'], edit)
    _relink(goc, d, x)
    with pytest.raises(GC.GateFileError, match='recomputed from the scorings'):
        GC.check_nguon(d, goc)


def test_round3_evidence_g4_and_classification_from_the_grid(tmp_path):
    d, goc = build_c8(tmp_path / 'a')
    x = d['o_c8'][0]['chuoi'][0]
    _rewrite(goc, x['file_cham'], lambda y: y['nums']['g4'].update(n_undef=1))
    _relink(goc, d, x, old=None)
    with pytest.raises(GC.GateFileError, match='recomputed from the scorings'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'b')
    x = d['o_c8'][0]['chuoi'][0]
    old = x['file_cham']['sha256']
    _rewrite(goc, x['file_cham'], lambda y: y['states'].pop('g4'))
    _relink(goc, d, x, old)
    with pytest.raises(GC.GateFileError, match='recomputed from the scorings|GC9-5'):
        GC.check_nguon(d, goc)


def test_round3_one_clip_h_comparison_per_part(tmp_path):
    """GC6-3 (round 3): two runs of one part of the session with two different comparisons are refused."""
    d, goc = build_c8(tmp_path)
    x = d['o_c8'][1]['chuoi'][0]
    alt = _jw(goc, 'drift_b.json', dict(json.load(open(os.path.join(goc, 'drift.json'), encoding='utf-8')), D_H_mm=1.5))
    x['file_so_h'] = alt
    with pytest.raises(GC.GateFileError, match='one per part'):
        GC.check_nguon(d, goc)


# ── Codex round 1 of code v6_C8 (GC6-7, 9, 10, 11, 15, 16) ────────────────────────────────────────────────────

def test_gc6_9_a_null_band_met_is_missing_whatever_the_constant_term():
    """Counter-example of Codex: F08, U_t 27 ms, a null band from 1.1686358576 m/s; with hang_m 60 mm the lower bound is
    above half the threshold at once, but a cell may still meet the null band: missing, never report-only-by-value."""
    for h in (0.002, 0.060):
        b = _blk('O1-F08', hang=h, bands_p=[dict(u_lo=0.0, u_hi=1.1686358576, gia_tri=0.0),
                                             dict(u_lo=1.1686358576, u_hi=5.0, gia_tri=None)])
        uh = GC.uhat_level('O1-F08', b, 0.027)
        assert uh['g2']['trang_thai'] == 'thieu' and uh['g2']['uhat'] is None, (h, uh['g2'])


def test_gc6_10_lower_bound_samples_inside_the_true_window(monkeypatch):
    """Counter-example of Codex: V(s) = max(s, 0), delta 1 ms, U_t just below 27 ms: every lower-bound value is
    reached by the true sup over [t - U_t, t + U_t] (a sample 27 ms away lies outside that window)."""
    sch = Schedule(FIG8['O1-F08'])

    def lin(sch_, t, a_m, amp=1.0):
        v = np.maximum(np.asarray(t, dtype=float), 0.0)
        z = np.zeros_like(v)
        return v, z, v, z

    monkeypatch.setattr(GC, 'schedule_samples', lin)
    u_t = float(np.nextafter(0.027, 0.0))
    blk = _blk(hang=0.0)
    tm = GC._Terms('g3', blk, u_t)
    lip = dict(L_V=1.0, L_W=0.0, L_u=1.0, L_r=0.0)
    m, mp, nc = GC.grid_size(u_t, -u_t, 0.05, 1e-3)
    assert mp == 26 and m == 27
    _, _, res = GC._grid_bounds(sch, [tm], lip, blk['a_m'], u_t, -u_t, 0.05, 1e-3)
    f_lo = res[0][2]
    t = -u_t + np.arange(nc + 1) * 1e-3
    t = t[(t >= 0.0) & (t <= 0.05)]
    assert np.all(f_lo <= u_t * (t + u_t) + 1e-18)


def test_gc6_10_arm_is_zero_where_the_schedule_stands_still():
    sch = Schedule(FIG8['O1-F05'])
    _, _, _, r = GC.schedule_samples(sch, np.array([-0.1, 0.0, sch.Tcmd, sch.Tcmd + 0.2]), 0.1445)
    assert np.all(r == 0.0)


def test_gc6_11_budget_covers_the_first_grid():
    b = _blk('O1-F08')
    with pytest.raises(GC.GateFileError, match='budget'):
        GC.uhat_level('O1-F08', b, 0.027, budget=1)
    with pytest.raises(GC.GateFileError, match='budget'):
        GC.uhat_level('O1-F08', b, 1e9)
    m, mp, nc = GC.grid_size(0.027, -0.027, 10.0, 1e-3)
    assert GC.grid_points(0.027, -0.027, 10.0, 1e-3) == nc + 2 * m + 3
    assert m * 1e-3 >= 0.027 and mp * 1e-3 + GC.EPS_T <= 0.027 and -0.027 + nc * 1e-3 >= 10.0


def test_gc6_7_g5_f08_needs_no_liam_choice_and_holds_off_the_gate_levels(tmp_path):
    uh = {c: dict(trang_thai='quyet') for c in GC.MASKABLE}
    uh['g5'] = dict(trang_thai='bien_nguong')
    assert GC.report_only_of('O1-F08', uh) == (('g5',), {'g5': ['liam_g5_f08']})
    with pytest.raises(GC.GateFileError, match='CAN LIAM QUYET'):          # elsewhere bien_nguong needs Liam
        GC.report_only_of('O1-F05', uh)
    with pytest.raises(GC.GateFileError, match='not needed'):
        GC.report_only_of('O1-F08', uh, {'g5': dict(chon='quyet')})
    gf = GC.read_gate_c8(_write(tmp_path, GC.fake_gate(('O1-F05',))))      # F08 not a gate level
    assert gf.required('O1-F08', 'HW', 'final', 'A-PID') == ('g1', 'g2', 'g3', 'g4')
    assert gf.required('O1-F08', 'SIM', 'script', 'B-chung') == o1_gate.CONDS
    assert gf.required('O1-K1', 'HW', 'final', 'A-PID') == o1_gate.CONDS


def test_ct2_6_ids_sha_is_the_comma_join():
    """CODE 19/11 (CT2-6): the frame id hash of HIEUCHUAN / CAM ids_sha, sha256 of '1,2' (no brackets)."""
    assert GC.sha256_ids([1, 2]) == hashlib.sha256(b'1,2').hexdigest()
    with pytest.raises(GC.GateFileError, match='whole numbers'):
        GC.sha256_ids(['1'])


def test_gc6_15_types_before_the_tolerance():
    assert not GC._close('0.002', 0.002) and not GC._close(True, 1.0) and GC._close(0.002, 0.002)
    d = _good()
    u = d['dieu_2b']['per_level']['O1-F05']['uhat']['g2']
    u['so_lan_tinh'] = float(u['so_lan_tinh'])
    with pytest.raises(GC.GateFileError, match='whole number'):
        GC.check(d)
    d = _good()
    d['dieu_2b']['per_level']['O1-F05']['uhat']['g2']['uhat'] = str(d['dieu_2b']['per_level']['O1-F05']['uhat']['g2']['uhat'])
    with pytest.raises(GC.GateFileError, match='numbers or null'):
        GC.check(d)


def test_gc6_16_reasons_of_the_mask_recomputed():
    d = _good()
    d['dieu_2b']['per_level']['O1-F08']['ly_do_report_only'] = {'g5': ['vuot_nua_nguong']}
    with pytest.raises(GC.GateFileError, match='ly_do_report_only'):
        GC.check(d)


def test_gc6_16_evidence_ii_iii_needs_every_set_of_numbers():
    """(ii), (iii) hold only when the property holds in BOTH sets of numbers (scored again with both calibrations)."""
    for k in ('e_do_xac_dinh_moi_nut', 'g4_can_huu_han_mien_hop_moi_nut'):
        d = _good()
        for r in d['runs_c8']['O1-F05']:
            if r['controller'] == 'LADRC':
                r['chung_cu']['bo_so'] = [dict(r['chung_cu']['bo_so'][0]), dict(r['chung_cu']['bo_so'][0], **{k: False})]
        with pytest.raises(GC.GateFileError, match='LADRC'):
            GC.check(_refresh(d))


def test_gc6_16_write_gate_on_a_boundary_case(tmp_path, monkeypatch):
    """A bien_nguong condition: write_gate refuses without Liam's choice, writes with it."""
    real = GC.uhat_level

    def tight(level, blk, u_t, budget=GC.BUDGET):
        return real(level, blk, u_t, budget=GC.grid_points(u_t, -u_t, Schedule(FIG8[level]).Tcmd + u_t, GC.DELTA0_S))
    monkeypatch.setattr(GC, 'uhat_level', tight)
    lv = 'O1-F05'
    hang = _straddle(lv)
    t = dict(a_m=0.1445, u_p=dict(hang_m=hang, he_so_r=0.0, u_phi_rad=0.0, hang_theo_u=[]),
             u_psi=dict(hang_rad=math.radians(0.1), hang_theo_u=[]))
    assert GC.uhat_level(lv, GC._level_terms(t, lv), 0.02)['g3']['trang_thai'] == 'bien_nguong'
    d = GC.fake_gate((lv,), u_t_s=0.02)
    runs = d['runs_c8'][lv]
    with pytest.raises(GC.GateFileError, match='CAN LIAM QUYET'):
        GC.level_block(lv, t['a_m'], t['u_p'], t['u_psi'], runs)
    q = {'g3': dict(chon='chi_bao_cao', ngay='2026-11-20', phien='local_x', nguyen_van='FAKE')}
    d['dieu_2b']['per_level'][lv] = GC.level_block(lv, t['a_m'], t['u_p'], t['u_psi'], runs, q)
    with pytest.raises(GC.GateFileError, match='CAN LIAM QUYET'):
        GC.write_gate(str(tmp_path / 'a.json'), d)
    assert not os.path.exists(str(tmp_path / 'a.json'))
    d['quyet_liam'] = {lv: q}
    GC.write_gate(str(tmp_path / 'b.json'), d)
    assert 'g3' in GC.read_gate_c8(str(tmp_path / 'b.json')).report_only(lv)


# ── Codex round 2 of code v6_C8 (GC6-1..6, 10, GC7-1, 2, 5; Liam at CODE "Sửa + vòng 3 ngay trên TK4") ──────────

def test_gc7_1_lower_bound_above_half_keeps_the_mask():
    """Counter-example of Codex round 2: F08, U_t 27 ms, hang_m 60 mm, a null band from 1.1686358585973085 m/s: the
    lower bound (60 mm) is above half the threshold, so G2, G3 report only whatever cells may still meet the band; the
    number is unknown (uhat null) and the reasons say so; no choice of Liam is asked."""
    b = _blk('O1-F08', hang=0.060, bands_p=[dict(u_lo=0.0, u_hi=1.1686358585973085, gia_tri=0.0),
                                            dict(u_lo=1.1686358585973085, u_hi=5.0, gia_tri=None)])
    uh = GC.uhat_level('O1-F08', b, 0.027)
    for c in ('g2', 'g3'):
        assert uh[c]['trang_thai'] == 'chi_bao_cao' and uh[c]['uhat'] is None and uh[c]['uhat_duoi'] > 0.05
    ro, why = GC.report_only_of('O1-F08', uh)
    assert ro == ('g2', 'g3', 'g5') and why['g2'] == ['vuot_nua_nguong', 'thieu_chua_phan_giai']


def test_gc6_10_margin_from_the_true_spacing(monkeypatch):
    """A Lipschitz-1 tent peaking between two grid points: every cell bound covers it with the true float spacing."""
    sch = Schedule(FIG8['O1-F08'])
    peak = 0.0155 + 0.37e-3

    def tent(sch_, t, a_m, amp=1.0):
        v = np.maximum(0.0, 0.01 - np.abs(np.asarray(t, dtype=float) - peak))
        z = np.zeros_like(v)
        return v, z, v, z

    monkeypatch.setattr(GC, 'schedule_samples', tent)
    blk = _blk(hang=0.0)
    tm = GC._Terms('g3', blk, 0.001)
    lip = dict(L_V=1.0, L_W=0.0, L_u=1.0, L_r=0.0)
    _, (ta, tb), res = GC._grid_bounds(sch, [tm], lip, blk['a_m'], 0.001, 0.0123456789, 0.03, 1e-3)
    for a, b, f in zip(ta, tb, res[0][0]):
        s = np.linspace(a - 0.001, b + 0.001, 401)
        assert f >= 0.001 * float(np.max(np.maximum(0.0, 0.01 - np.abs(s - peak))))


@pytest.mark.parametrize('band, msg', [
    (dict(u_lo_m_s=0.0, u_hi_m_s=9.0, P4_m=0.0, A4_deg='invalid', trang_thai='dat'), 'GC7-2'),
    (dict(u_lo_m_s=0.0, u_hi_m_s=9.0, P4_m=0.0, A4_deg=0.1, trang_thai=[]), 'GC7-2'),
    (dict(u_lo_m_s=0.0, u_hi_m_s=9.0, P4_m=None, A4_deg=0.1, trang_thai='dat'), 'GC7-2'),
    (dict(u_lo_m_s='0', u_hi_m_s=9.0, P4_m=0.0, A4_deg=0.1, trang_thai='dat'), 'GC7-2'),
])
def test_gc7_2_bands_of_the_lock_checked(tmp_path, band, msg):
    d, goc = build_c8(tmp_path)
    _rewrite_khoa(goc, d, lambda k: k['muc']['O1-F05'].update(hang_theo_u=[band]))
    with pytest.raises(GC.GateFileError, match=msg):
        GC.check_nguon(d, goc)


def test_gc7_5_shutter_table_on_the_gate_levels_and_k(tmp_path):
    d, goc = build_c8(tmp_path / 'a', levels=('O1-F05',))
    _rewrite_khoa(goc, d, lambda k: k['bang_man_trap'].update({'O1-F02': dict(nhanh='mot_khung'),
                                                                       'O1-F08': dict(nhanh='mot_khung')}))
    GC.check_nguon(d, goc)                                             # the lock has every level, the gate one
    _rewrite_khoa(goc, d, lambda k: k['bang_man_trap'].pop('K'))
    with pytest.raises(GC.GateFileError, match='GC7-5'):
        GC.check_nguon(d, goc)


def test_gc6_1_one_esp32_log_per_physical_run(tmp_path):
    d, goc = build_c8(tmp_path)
    x, y = d['o_c8'][0]['chuoi'][0], d['o_c8'][1]['chuoi'][0]
    esp_x = json.load(open(os.path.join(goc, x['file_gio']['duong_dan']), encoding='utf-8'))['sha256']['esp32']

    def same_log(g):
        g['sha256']['esp32'] = esp_x
        g['nguon'][1]['sha256'] = esp_x
    _rewrite(goc, y['file_gio'], same_log)
    _relink(goc, d, y)
    with pytest.raises(GC.GateFileError, match='ESP32 log of'):
        GC.check_nguon(d, goc)


def test_gc6_3_the_drift_of_the_run_part(tmp_path):
    d, goc = build_c8(tmp_path / 'a')                                 # the comparison (H in the middle, H at the end)
    _rewrite(goc, d['o_c8'][0]['chuoi'][0]['file_so_h'],
             lambda y: y.update(s0_session_sha256=d['tien_de_c8']['clip_h']['giua']['sha256'],
                                s1_session_sha256=d['tien_de_c8']['clip_h']['cuoi']['sha256']))
    for c in d['o_c8']:
        c['chuoi'][0]['file_so_h']['sha256'] = d['o_c8'][0]['chuoi'][0]['file_so_h']['sha256']
        x = c['chuoi'][0]                                             # CODE 20/11: luoi.json names that drift file too
        lu = json.load(open(os.path.join(goc, x['file_cham']['duong_dan']), encoding='utf-8'))
        _relink_nodes(goc, d, x, lu, file_so_h=dict(x['file_so_h']))
    with pytest.raises(GC.GateFileError, match='first part'):         # every run is before the middle clip
        GC.check_nguon(d, goc)
    _csv_edit(goc, d, lambda ls: ls[:2] + [ls[-1]] + ls[2:-1])         # now every run is after it
    GC.check_nguon(d, goc)
    # GC9-3 (round 4): vuot recomputed with the gate thresholds (eps_H 1.0 mm, 0.2 deg), position and heading, never the
    # threshold the drift file states; a missing quantity is vuot (TC3-10)
    for name, edit in [('b', dict(D_H_vi_tri_mm=100.0)),
                       ('b2', dict(D_H_vi_tri_mm=3.0, eps_H_mm=4.0)),          # Codex: 3 mm under the drift's 4 mm
                       ('b3', dict(D_H_huong_deg=0.5)),                        # the heading branch of J section 5
                       ('b4', dict(trang_thai_huong='thieu', D_H_huong_deg=None))]:
        d, goc = build_c8(tmp_path / name)
        _rewrite(goc, d['o_c8'][0]['chuoi'][0]['file_so_h'], lambda y: y.update(edit))
        for c in d['o_c8']:
            c['chuoi'][0]['file_so_h']['sha256'] = d['o_c8'][0]['chuoi'][0]['file_so_h']['sha256']
        with pytest.raises(GC.GateFileError, match='GC9-3|vuot_do_thieu'):
            GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'c')                                 # an old drift file (v1) is refused
    _rewrite(goc, d['o_c8'][0]['chuoi'][0]['file_so_h'], lambda y: y.pop('version'))
    for c in d['o_c8']:
        c['chuoi'][0]['file_so_h']['sha256'] = d['o_c8'][0]['chuoi'][0]['file_so_h']['sha256']
    with pytest.raises(GC.GateFileError, match='c7_calib_drift_v2'):
        GC.check_nguon(d, goc)


def test_gc6_3_each_scoring_with_its_edo_file(tmp_path):
    d, goc = build_c8(tmp_path / 'a')
    x = d['o_c8'][0]['chuoi'][0]
    _rewrite(goc, x['file_chung_cu'], lambda y: y['nguon'][1].update(sha256='a' * 64))   # an edo sha of nothing
    with pytest.raises(GC.GateFileError, match='e_do files'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'b')
    x = d['o_c8'][0]['chuoi'][0]
    _rewrite(goc, x['file_edo'], lambda y: y.update(gio_sha256='b' * 64))
    _rewrite(goc, x['file_chung_cu'], lambda y: y['nguon'][1].update(sha256=x['file_edo']['sha256']))
    with pytest.raises(GC.GateFileError, match='not made from the time file'):
        GC.check_nguon(d, goc)


def test_gc6_6_node_scores_tied_to_the_grid(tmp_path):
    for name, edit, msg in [
            ('a', lambda k, y: y['khung_tuong_doi_v1']['mien_phai_phu'].update(D_s=0.5), 'D_s = nodes_s'),
            ('b', lambda k, y: y['khung_tuong_doi_v1'].update(gio_sha256='c' * 64), 'not of the time file'),
            ('c', lambda k, y: y.update(outcome='hong_he_do'), 'hong_nut')]:   # lost by its score, not the grid
        d, goc = build_c8(tmp_path / name)
        x = d['o_c8'][0]['chuoi'][0]
        p = os.path.join(goc, x['file_cham']['duong_dan'])
        lu = json.load(open(p, encoding='utf-8'))
        r = dict(duong_dan=lu['nguon_nut'][1]['duong_dan'], sha256=lu['nguon_nut'][1]['sha256'])
        if name == 'c':
            _rewrite(goc, r, lambda y: (y.pop('khung_tuong_doi_v1'), y.update(outcome='hong_he_do')))
        else:
            _rewrite(goc, r, lambda y: edit(1, y))
        lu['nguon_nut'][1]['sha256'] = r['sha256']
        old = x['file_cham']['sha256']
        _rewrite(goc, x['file_cham'], lambda y: y.update(nguon_nut=lu['nguon_nut']))
        _rewrite(goc, x['file_chung_cu'], lambda y: y['nguon'][0].update(sha256=x['file_cham']['sha256']))
        for m in d['camera']['c8b']['moc_dich']['nguon']['manifests']:
            if json.load(open(os.path.join(goc, m['duong_dan']), encoding='utf-8'))['file_cham_sha256'] == old:
                _rewrite(goc, m, lambda y: y.update(file_cham_sha256=x['file_cham']['sha256']))
        with pytest.raises(GC.GateFileError, match=msg):
            GC.check_nguon(d, goc)


@pytest.mark.parametrize('lost_score, msg', [
    ({'outcome': 'hong_he_do'}, 'task'),                                  # Codex round 3: a bare lost score
    ({'outcome': 'hong_he_do', 'task': 'O1-F02', 'esp32_sha256': 'a' * 64, 'frames_sha256': 'b' * 64}, 'another log'),
])
def test_round3_lost_node_tied_to_its_run(tmp_path, lost_score, msg):
    """GC6-6 (round 3): a node lost on the grid is dropped from the domain only when its score is of this run."""
    d, goc = build_c8(tmp_path)
    x = d['o_c8'][0]['chuoi'][0]
    lu = json.load(open(os.path.join(goc, x['file_cham']['duong_dan']), encoding='utf-8'))
    r = _jw(goc, 'luot/bare_lost.json', lost_score)
    lu['nguon_nut'][1].update(r)
    old = x['file_cham']['sha256']
    _rewrite(goc, x['file_cham'], lambda y: y.update(nguon_nut=lu['nguon_nut'], hong_nut=[False, True]))
    _rewrite(goc, x['file_chung_cu'], lambda y: y['nguon'][0].update(sha256=x['file_cham']['sha256']))
    for m in d['camera']['c8b']['moc_dich']['nguon']['manifests']:
        if json.load(open(os.path.join(goc, m['duong_dan']), encoding='utf-8'))['file_cham_sha256'] == old:
            _rewrite(goc, m, lambda y: y.update(file_cham_sha256=x['file_cham']['sha256'],
                                                mien_da_tinh=y['mien_da_tinh'][:1]))
    with pytest.raises(GC.GateFileError, match=msg):
        GC.check_nguon(d, goc)


@pytest.mark.parametrize('edit', [
    lambda s: s.update(fit_ids=[None], marks=[None]),                     # Codex round 3
    lambda s: s.update(fit_ids=[1, 1, 2]),
    lambda s: s.update(fit_ids=[1, 2, 9]),                                # a fit id without its mark
    lambda s: s['ref_corners_px'].update({'2': [[0, 0], [1, 0]]}),        # not four corners
    lambda s: s['marks'].update({'3': [0.0, None]})])
def test_round3_session_files_as_kiemcheo_reads_them(tmp_path, edit):
    d, goc = build_c8(tmp_path)
    m = d['tien_de_c8']['clip_h']['dau'][1]
    _rewrite(goc, m, edit)
    _rewrite(goc, d['tien_de_c8']['file'], lambda x: x['sha256']['sessions'].update(H_b=m['sha256']))
    with pytest.raises(GC.GateFileError, match='camera session file'):
        GC.check_nguon(d, goc)


# ── ghep: the gate assembled from its sources (CODE 19/11; Liam at CODE "gộp rồi chạy ngay thên TK4") ─────────────

GHEP_KEYS = ('levels', 'muc_kha_thi', 'u_t_branch', 'bo_cham', 'ke_hoach_c8', 'thu_tu_clip', 'khoa_c8b', 'o_c8',
             'runs_c8', 'muc_loai_e_do', 'chung_cu_muc', 'dieu_2b', 'camera', 'tien_de_c8')


BANG_CHUNG = dict(khoa_kiem_truoc_giu_ngoai='c8b_khoa_v5.json', kiem_yaw_theo_hieu_mien_quyet='chuan_u/c8b_a6_o.json',
                  kiem_vi_tri_rieng='chuan_u/c8b_kiem_vi_tri.json', luat_giu_mask_theo_phan_do='c8b_khoa_v5.json')


def _ghep(goc, ho_so='chuan_u/ho_so_ghep.json', **kw):
    return GC.ghep(goc, 'ke_hoach_c8.json', 'thu_tu_clip.csv', 'c8b_khoa_v5.json', 'tien_de_c8.json', 'kiem_cheo.json',
                   dict(BANG_CHUNG), ho_so, **kw)


@pytest.mark.parametrize('two, lost', [(False, False), (True, False), (False, True)])
def test_ghep_rebuilds_the_gate_from_its_sources(tmp_path, two, lost):
    d, goc = build_c8(tmp_path, two=two)
    if lost:                                       # a run lost to an operation (boot), then its rerun
        _add_lost(goc, d, 0, 1, ['thao_tac'], ops=('boot',))
    for c in d['o_c8']:                            # ghep names a run by the buoc of its row
        for j, x in enumerate(c['chuoi']):
            x['run_id'] = c['o'] + (f'_lai{j}' if j else '')
    g = _ghep(goc)
    for k in ('o_c8',):
        for cg, cd in zip(g[k], d[k]):
            assert cg['trang_thai'] == cd['trang_thai'] and len(cg['chuoi']) == len(cd['chuoi'])
            for eg, ed in zip(cg['chuoi'], cd['chuoi']):
                assert {q: v for q, v in eg.items() if q not in ('run_id', 'file_log', 'file_thao_tac')} == \
                    {q: v for q, v in ed.items() if q not in ('run_id', 'file_log', 'file_thao_tac')}
                for q in ('file_log', 'file_thao_tac'):
                    assert (eg[q] is None) == (ed[q] is None) and (eg[q] is None or eg[q]['sha256'] == ed[q]['sha256'])
    for k in ('levels', 'muc_kha_thi', 'u_t_branch', 'bo_cham', 'ke_hoach_c8', 'thu_tu_clip', 'khoa_c8b',
              'muc_loai_e_do', 'dieu_2b', 'camera', 'tien_de_c8'):
        if k == 'camera':                          # the manifests in another order (by sha256)
            for x in (g, d):
                x[k]['c8b']['moc_dich']['nguon']['manifests'].sort(key=lambda m: m['sha256'])
        assert g[k] == d[k], k
    assert {lv: [r['controller'] for r in rs] for lv, rs in g['runs_c8'].items()} == \
        {lv: [r['controller'] for r in rs] for lv, rs in d['runs_c8'].items()}
    p = str(tmp_path / 'gate.json')
    sha = GC.write_gate(p, g, goc)                                      # both checks, then written once
    gf = GC.read_gate_c8(p)
    assert gf.sha256 == sha and not gf.fake and not gf.fake_sign
    hs = json.load(open(os.path.join(goc, 'chuan_u/ho_so_ghep.json'), encoding='utf-8'))
    assert hs['format'] == GC.HO_SO_FORMAT and set(hs['cham_luc']) == {x['run_id'] for c in g['o_c8'] for x in c['chuoi']}
    assert _ghep(goc) == g                                              # the same dossier bytes: run again


def test_ghep_cli_and_refusals(tmp_path):
    d, goc = build_c8(tmp_path)
    out = str(tmp_path / 'gate.json')
    args = ['ghep', '--goc', goc, '--ke-hoach', 'ke_hoach_c8.json', '--thu-tu-clip', 'thu_tu_clip.csv',
            '--khoa-c8b', 'c8b_khoa_v5.json', '--tien-de', 'tien_de_c8.json', '--kiem-cheo', 'kiem_cheo.json',
            '--ho-so', 'chuan_u/ho_so_cli.json', '--out', out] + \
        sum((['--bang-chung', f'{k}={p}'] for k, p in BANG_CHUNG.items()), [])
    assert GC.main(args) == 0 and GC.read_gate_c8(out).levels == list(LEVELS)
    assert GC.main(args) == 1                                           # the gate file is never replaced
    # the C8b lock without numbers for a closed term: C8 is not locked
    d, goc = build_c8(tmp_path / 'b')
    _rewrite_khoa(goc, d, lambda k: k['muc']['O1-F05']['hang_dong'].update(hang_m=None))
    with pytest.raises(GC.GateFileError, match='closed term'):
        _ghep(goc)
    # a dossier already there with other bytes is never replaced
    d, goc = build_c8(tmp_path / 'c')
    with open(os.path.join(goc, 'chuan_u/ho_so_ghep.json'), 'w', encoding='utf-8') as f:
        f.write('{}')
    with pytest.raises(GC.GateFileError, match='never replaced'):
        _ghep(goc)
    # the U manifests disagree on their source block
    d, goc = build_c8(tmp_path / 'e')
    _rewrite(goc, d['camera']['c8b']['moc_dich']['nguon']['manifests'][0], lambda y: y['nguon'].update(tap_moc='0' * 64))
    with pytest.raises(GC.GateFileError, match='one nguon'):
        _ghep(goc)
    # above eps_H without the second scoring
    d, goc = build_c8(tmp_path / 'f')
    _rewrite(goc, d['o_c8'][0]['chuoi'][0]['file_so_h'], lambda y: y.update(vuot=True, D_H_vi_tri_mm=3.0))
    old = d['o_c8'][0]['chuoi'][0]['file_cham']['sha256']
    _rewrite(goc, d['o_c8'][0]['chuoi'][0]['file_cham'],
             lambda y: y.update(file_so_h=d['o_c8'][0]['chuoi'][0]['file_so_h']))
    with pytest.raises(GC.GateFileError, match='second scoring is due'):
        _ghep(goc)


# ── round 4 (Codex lượt chung 01a1f4d8; Liam qua CODE 20/11 "sửa phần độc lập ngay") ─────────────────────────────

def _relink_nodes(goc, d, x, lu, **luoi_kw):
    """luoi.json of run x rewritten with its node list lu['nguon_nut'] (and luoi_kw); evidence file and manifest tied."""
    old = x['file_cham']['sha256']
    _rewrite(goc, x['file_cham'], lambda y: y.update(nguon_nut=lu['nguon_nut'], **luoi_kw))
    _rewrite(goc, x['file_chung_cu'], lambda y: y['nguon'][0].update(sha256=x['file_cham']['sha256']))
    for m in d['camera']['c8b']['moc_dich']['nguon']['manifests']:
        if json.load(open(os.path.join(goc, m['duong_dan']), encoding='utf-8'))['file_cham_sha256'] == old:
            _rewrite(goc, m, lambda y: y.update(file_cham_sha256=x['file_cham']['sha256']))


def test_round4_c8b_absolute_paths_and_missing_node_scores(tmp_path):
    """GC9-1: c8b_luoi names node scores by absolute path (inside the C8 folder: accepted) and only the nodes with a
    score file; a path outside the folder is still refused."""
    d, goc = build_c8(tmp_path / 'a')
    x = d['o_c8'][0]['chuoi'][0]
    lu = json.load(open(os.path.join(goc, x['file_cham']['duong_dan']), encoding='utf-8'))
    for e in lu['nguon_nut']:
        e['duong_dan'] = os.path.abspath(os.path.join(goc, e['duong_dan']))
    _relink_nodes(goc, d, x, lu)
    GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'b')                                  # node 1 without a score file: not scored
    x = d['o_c8'][0]['chuoi'][0]
    lu = json.load(open(os.path.join(goc, x['file_cham']['duong_dan']), encoding='utf-8'))
    lu['nguon_nut'] = lu['nguon_nut'][:1]
    _relink_nodes(goc, d, x, lu, nodes_ok=[True, False])
    for m in d['camera']['c8b']['moc_dich']['nguon']['manifests']:
        if json.load(open(os.path.join(goc, m['duong_dan']), encoding='utf-8'))['file_cham_sha256'] == x['file_cham']['sha256']:
            _rewrite(goc, m, lambda y: y.update(mien_da_tinh=y['mien_da_tinh'][:1]))
    with pytest.raises(GC.GateFileError, match='finite G4|recomputed|node states'):     # passes the node checks
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'c')
    x = d['o_c8'][0]['chuoi'][0]
    lu = json.load(open(os.path.join(goc, x['file_cham']['duong_dan']), encoding='utf-8'))
    outside = tmp_path / 'c' / 'nut_out.json'
    outside.write_text(json.dumps(json.load(open(os.path.join(goc, lu['nguon_nut'][0]['duong_dan']), encoding='utf-8'))),
                       encoding='utf-8')
    lu['nguon_nut'][0].update(duong_dan=str(outside), sha256=GC.sha256_file(str(outside)))
    _relink_nodes(goc, d, x, lu)
    with pytest.raises(GC.GateFileError, match='inside it'):
        GC.check_nguon(d, goc)


def test_round4_c8b_nguon_3b_absolute_path(tmp_path):
    d, goc = build_c8(tmp_path)
    src = _jw(goc, '3b/3b_F05.json', dict(version='c8b_3b'))
    d['bo_cham']['bang_man_trap']['O1-F05'] = dict(nhanh='alpha', nguon_3b=dict(src))
    _rewrite_khoa(goc, d, lambda k: k['bang_man_trap'].update({'O1-F05': dict(
        nhanh='alpha', nguon_3b=dict(duong_dan=os.path.abspath(os.path.join(goc, src['duong_dan'])),
                                     sha256=src['sha256']))}))
    GC.check_nguon(d, goc)


def test_round4_scoring_times_from_the_node_scorer_states(tmp_path):
    """GC6-1 (C8B 19/11 khuya): a run is first scored when its first node is (scorer.nut[k].luc_chay), not when
    luoi.json is written; a node scored before the C8b lock is refused."""
    d, goc = build_c8(tmp_path)
    x = d['o_c8'][0]['chuoi'][0]
    old = x['file_cham']['sha256']
    _rewrite(goc, x['file_cham'], lambda y: y['scorer'].update(
        nut=[dict(commit='5' * 40, dirty=False, luc_chay='2026-11-21T07:30:00+07:00')]))
    _relink(goc, d, x, old)
    with pytest.raises(GC.GateFileError, match='not before the first scoring of a C8 run'):
        GC.check_nguon(d, goc)


def test_round4_node_without_score_not_ok_and_lost_run_log(tmp_path):
    """GC6-6: a node without a score file kept as ok is refused; GC6-1 (round 4): a lost run without a time file still
    carries the log thu_tu_clip.csv names."""
    d, goc = build_c8(tmp_path / 'a')
    x = d['o_c8'][0]['chuoi'][0]
    lu = json.load(open(os.path.join(goc, x['file_cham']['duong_dan']), encoding='utf-8'))
    lu['nguon_nut'] = lu['nguon_nut'][:1]
    _relink_nodes(goc, d, x, lu)                                       # nodes_ok stays [True, True]
    with pytest.raises(GC.GateFileError, match='no score file'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'b')
    x = _add_lost(goc, d, 0, 1, ['he_do'], gio=False, d1='khong_kiem')
    GC.check_nguon(d, goc)
    x['file_log'] = None
    with pytest.raises(GC.GateFileError, match='carries it'):
        GC.check_nguon(d, goc)


def test_round4_held_out_is_every_run(tmp_path):
    d, goc = build_c8(tmp_path)
    _rewrite(goc, d['chuan_u']['ho_so_c8'], lambda y: y.update(giu_ngoai=y['giu_ngoai'][1:]))
    with pytest.raises(GC.GateFileError, match='GC9-2'):
        GC.check_nguon(d, goc)


def test_round4_chain_starts_with_the_run_itself(tmp_path):
    """GC9-4 (Codex): a cell whose only row is <run>_lai2 is refused."""
    d, goc = build_c8(tmp_path)
    _csv_edit(goc, d, lambda ls: [ls[0], ls[1], ls[2].replace(',O1-F02_PID,', ',O1-F02_PID_lai2,')] + ls[3:])
    with pytest.raises(GC.GateFileError, match='GC9-4'):
        GC.check_nguon(d, goc)


def test_round4_grid_outcome_from_its_states(tmp_path):
    """GC9-5 (Codex): five states dat but outcome khong_dat is refused; report_only conditions do not decide."""
    d, goc = build_c8(tmp_path / 'a')
    x = d['o_c8'][0]['chuoi'][0]
    old = x['file_cham']['sha256']
    _rewrite(goc, x['file_cham'], lambda y: y.update(outcome='khong_dat'))
    _relink(goc, d, x, old)
    with pytest.raises(GC.GateFileError, match='GC9-5'):
        GC.check_nguon(d, goc)
    fc = dict(trang_thai_luot='cham', outcome='dat', report_only=['g5'],
              states=dict(g1='dat', g2='dat', g3='dat', g4='dat', g5='khong_dat'))
    GC._grid_outcome(fc, [], 'x')
    with pytest.raises(GC.GateFileError, match='GC9-5'):
        GC._grid_outcome(dict(fc, report_only=[]), [], 'x')


def test_round4_lost_node_offset_and_nodes_ok(tmp_path):
    """GC6-6 (round 4): a lost node score with the node time of another node, or a lost node kept as ok, is refused."""
    for name, edit, ok, msg in [('a', lambda y: y.update(offset_s=0.0), [True, False], 'offset_s'),
                                ('b', lambda y: None, [True, True], 'nodes_ok')]:
        d, goc = build_c8(tmp_path / name)
        x = d['o_c8'][0]['chuoi'][0]
        lu = json.load(open(os.path.join(goc, x['file_cham']['duong_dan']), encoding='utf-8'))
        r = dict(duong_dan=lu['nguon_nut'][1]['duong_dan'], sha256=lu['nguon_nut'][1]['sha256'])
        _rewrite(goc, r, lambda y: (y.pop('khung_tuong_doi_v1'), y.update(outcome='hong_he_do'), edit(y)))
        lu['nguon_nut'][1]['sha256'] = r['sha256']
        _relink_nodes(goc, d, x, lu, hong_nut=[False, True], nodes_ok=ok, loai_hong=['he_do'])
        with pytest.raises(GC.GateFileError, match=msg):
            GC.check_nguon(d, goc)


def test_round4_session_fields_kiemcheo_reads(tmp_path):
    """GC6-4 (round 4): width, height, chosen = model_khoa, as c7_kiem_cheo reads the session."""
    for name, edit in [('a', lambda s: s.pop('width')), ('b', lambda s: s.update(model_khoa='other')),
                       ('c', lambda s: s.pop('chosen'))]:
        d, goc = build_c8(tmp_path / name)
        m = d['tien_de_c8']['clip_h']['dau'][1]
        _rewrite(goc, m, edit)
        _rewrite(goc, d['tien_de_c8']['file'], lambda x: x['sha256']['sessions'].update(H_b=m['sha256']))
        with pytest.raises(GC.GateFileError, match='camera session file'):
            GC.check_nguon(d, goc)


def test_round4_ghep_ignores_copies_and_cleans_its_dossier(tmp_path):
    """GC9-7: a second e_do of the same time file and a copy of the S7 manifest do not change the gate; GC9-8: the
    dossier path is checked before writing and taken back when the gate is refused."""
    d, goc = build_c8(tmp_path / 'a')
    x = d['o_c8'][0]['chuoi'][0]
    e = json.load(open(os.path.join(goc, x['file_edo']['duong_dan']), encoding='utf-8'))
    _jw(goc, 'aaa/edo_copy.json', dict(e, trang_thai='chua_ket_luan'))
    with open(os.path.join(goc, 'khoa_s7.json'), 'rb') as f:
        raw = f.read()
    os.makedirs(os.path.join(goc, 'aaa'), exist_ok=True)
    with open(os.path.join(goc, 'aaa', 'khoa_s7_copy.json'), 'wb') as f:
        f.write(raw)
    _rewrite(goc, dict(duong_dan='kiem_cheo.json', sha256=GC.sha256_file(os.path.join(goc, 'kiem_cheo.json'))),
             lambda y: y['khoa_s7'].update(file=os.path.abspath(os.path.join(goc, 'khoa_s7.json'))))
    kc_sha = GC.sha256_file(os.path.join(goc, 'kiem_cheo.json'))
    _rewrite(goc, d['tien_de_c8']['file'], lambda y: y['sha256'].update(kiem_cheo=kc_sha))
    g = _ghep(goc)
    assert g['o_c8'][0]['chuoi'][0]['file_edo'] == x['file_edo']
    GC.write_gate(str(tmp_path / 'g.json'), g, goc)
    d, goc = build_c8(tmp_path / 'b')
    with pytest.raises(GC.GateFileError, match='inside it'):
        _ghep(goc, ho_so='../outside.json')
    assert not os.path.exists(os.path.join(goc, '..', 'outside.json'))
    _rewrite(goc, d['tien_de_c8']['file'], lambda y: y.update(ket_qua='khong_dat'))     # the gate is refused
    with pytest.raises(GC.GateFileError, match='CAN LIAM QUYET'):
        GC.ghep_va_ghi(str(tmp_path / 'b.json'), goc, 'ke_hoach_c8.json', 'thu_tu_clip.csv', 'c8b_khoa_v5.json',
                       'tien_de_c8.json', 'kiem_cheo.json', dict(BANG_CHUNG), 'chuan_u/ho_so_new.json')
    assert not os.path.exists(os.path.join(goc, 'chuan_u', 'ho_so_new.json'))


# ── round 5 (Codex lượt chung 2 01a1f518; Liam qua CODE 20/11 "Sửa ngay, Codex sau dữ liệu thật") ──────────────

def _rewrite_nodes(goc, d, x, edit, h1=False):
    """Every node score of a scoring of run x rewritten by edit; luoi.json and what names it relinked."""
    key = 'file_cham_h1' if h1 else 'file_cham'
    lu = json.load(open(os.path.join(goc, x[key]['duong_dan']), encoding='utf-8'))
    for e in lu['nguon_nut']:
        r = dict(duong_dan=e['duong_dan'], sha256=e['sha256'])
        _rewrite(goc, r, edit)
        e['sha256'] = r['sha256']
    old = x[key]['sha256']
    _rewrite(goc, x[key], lambda y: y.update(nguon_nut=lu['nguon_nut']))
    _relink(goc, d, x, old, h1=h1)


def test_round5_m2_window_before_the_boot(tmp_path):
    """GC6-2: du_cua_so only when the window of the check ends before the boot (c8b_gio f11)."""
    tb = dict(co_boot=True, t_boot_log_s=5.0, du_cua_so=True, trang_thai='truot', thao_tac=['boot'])
    d, goc = build_c8(tmp_path)
    _add_lost(goc, d, 0, 1, ['thao_tac', 'he_do'], d1='truot', dem=True, ops=('boot',), luoi_kw=dict(loai_hong=['thao_tac']),
              gio_kw=dict(ok=False, ket_luan='hong_he_do', dem_truot_kiem_gio=True, cua_so_ghep_log_s=[1.0, 9.0],
                          kiem_gio_d1=dict(trang_thai='truot', truoc_boot=tb)))
    with pytest.raises(GC.GateFileError, match='without a boot time'):
        GC.check_nguon(d, goc)


def test_round5_h1_scoring_with_the_h1_session(tmp_path):
    """GC6-3 (CODE 20/11): the H1 node scores carry session_sha256 = s1_session_sha256 of the comparison."""
    d, goc = build_c8(tmp_path, two=True)
    x = d['o_c8'][0]['chuoi'][0]
    s0 = d['tien_de_c8']['clip_h']['dau'][0]['sha256']
    _rewrite_nodes(goc, d, x, lambda y: y.update(session_sha256=s0), h1=True)       # the H0 calibration again
    with pytest.raises(GC.GateFileError, match='GC6-3'):
        GC.check_nguon(d, goc)


def test_round5_one_locked_model_and_the_premise_kiem_cheo(tmp_path):
    """GC6-4: the four clip H sessions of one model (and the premise model); GC10-5: the premise names the kiem_cheo of
    the dossier."""
    d, goc = build_c8(tmp_path / 'a')
    m = d['tien_de_c8']['clip_h']['cuoi']
    _rewrite(goc, m, lambda s: s.update(chosen='other', model_khoa='other'))
    _rewrite(goc, d['tien_de_c8']['file'], lambda x: x['sha256']['sessions'].update(H_z=m['sha256']))
    with pytest.raises(GC.GateFileError, match='GC6-3|GC6-4|part of the session'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'b')
    _rewrite(goc, d['tien_de_c8']['file'], lambda x: x.update(model='polynomial'))
    with pytest.raises(GC.GateFileError, match='GC6-4'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'c')
    _rewrite(goc, d['tien_de_c8']['file'], lambda x: x['sha256'].update(kiem_cheo='a' * 64))
    with pytest.raises(GC.GateFileError, match='GC10-5'):
        GC.check_nguon(d, goc)


def test_round5_grid_nodes_of_its_time_file(tmp_path):
    """GC6-6: luoi.nodes_s is nodes_s of the time file (a grid keeping only node 0 of two)."""
    d, goc = build_c8(tmp_path)
    x = d['o_c8'][0]['chuoi'][0]
    _rewrite(goc, x['file_gio'], lambda y: y.update(nodes_s=[0.0, 0.001, 0.002]))
    _relink(goc, d, x)
    with pytest.raises(GC.GateFileError, match='GC6-6'):
        GC.check_nguon(d, goc)


def test_round5_grid_report_only_and_states_from_nodes(tmp_path):
    """GC9-5: report_only of a C8 grid is that of J (Codex: F05 with report_only g2 and outcome dat); a grid state dat
    needs every node dat; G1 by g1_rule."""
    for name, edit, msg in [
            ('a', lambda y: (y.update(report_only=['g2'], outcome='dat'), y['states'].update(g2='khong_dat')), 'report_only'),
            ('b', lambda y: (y['states'].update(g1='khong_dat'), y.update(outcome='khong_dat')), 'node states')]:
        d, goc = build_c8(tmp_path / name)
        x = next(c for c in d['o_c8'] if c['muc'] == 'O1-F05')['chuoi'][0]
        old = x['file_cham']['sha256']
        _rewrite(goc, x['file_cham'], edit)
        _relink(goc, d, x, old)
        with pytest.raises(GC.GateFileError, match=msg):
            GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'c')                                  # a node khong_dat, the grid dat: refused
    x = d['o_c8'][0]['chuoi'][0]
    _rewrite_nodes(goc, d, x, lambda y: y['gate_v3']['g3'].update(state='khong_dat'))
    with pytest.raises(GC.GateFileError, match='node states'):
        GC.check_nguon(d, goc)


def test_round5_u_t_of_the_grid(tmp_path):
    """GC10-3: u_max_s is the largest node U_t (Codex: u_t_nodes [0.007, 0.040], u_max_s 0.007); GC9-6 (Liam 20/11):
    a grid without U_t at a node leaves the run chua_ket_luan and its cell without an eligible run."""
    d, goc = build_c8(tmp_path / 'a')
    x = d['o_c8'][0]['chuoi'][0]
    lu = json.load(open(os.path.join(goc, x['file_cham']['duong_dan']), encoding='utf-8'))
    r = dict(duong_dan=lu['nguon_nut'][1]['duong_dan'], sha256=lu['nguon_nut'][1]['sha256'])
    _rewrite(goc, r, lambda y: y.update(U_t_s=0.040))
    lu['nguon_nut'][1]['sha256'] = r['sha256']
    old = x['file_cham']['sha256']
    _rewrite(goc, x['file_cham'], lambda y: y.update(nguon_nut=lu['nguon_nut'], u_t_nodes=[0.007, 0.040]))
    _relink(goc, d, x, old)
    with pytest.raises(GC.GateFileError, match='GC10-3'):
        GC.check_nguon(d, goc)
    # GC9-6: the chain flag and the cell state follow the grid
    g = _good()
    c = g['o_c8'][0]
    c['chuoi'][0]['thieu_u_t'] = True
    with pytest.raises(GC.GateFileError, match='GC9-6'):
        GC.check(g)                                                   # dat with thieu_u_t
    c['chuoi'][0]['ket_qua'] = 'chua_ket_luan'
    with pytest.raises(GC.GateFileError, match='recounted'):
        GC.check(g)                                                   # the cell is thieu_u_t
    assert GC.count_chain(c['chuoi'])[0] == 'co_luot'


def test_round5_ghep_run_without_u_t_is_chua_ket_luan(tmp_path):
    """GC9-6 through ghep: a grid lacking U_t at node 1; ghep marks the run, the cell has no eligible run, and the
    level then misses its controller's run (C8 not done there), as Liam chose."""
    d, goc = build_c8(tmp_path)
    x = d['o_c8'][0]['chuoi'][0]
    lu = json.load(open(os.path.join(goc, x['file_cham']['duong_dan']), encoding='utf-8'))
    r = dict(duong_dan=lu['nguon_nut'][1]['duong_dan'], sha256=lu['nguon_nut'][1]['sha256'])
    _rewrite(goc, r, lambda y: y.pop('U_t_s'))
    lu['nguon_nut'][1]['sha256'] = r['sha256']
    old = x['file_cham']['sha256']
    _rewrite(goc, x['file_cham'], lambda y: y.update(nguon_nut=lu['nguon_nut'], u_t_nodes=[0.007, None], u_max_s=None))
    _relink(goc, d, x, old)
    g = _ghep(goc)
    c = g['o_c8'][0]
    assert c['trang_thai'] == 'thieu_u_t' and c['chuoi'][0]['thieu_u_t'] and c['chuoi'][0]['ket_qua'] == 'chua_ket_luan'
    with pytest.raises(GC.GateFileError, match='no eligible run of'):
        GC.write_gate(str(tmp_path / 'g.json'), g, goc)


def test_round5_scorer_from_its_node_states(tmp_path):
    """GC10-2 (Codex): scorer commit A clean, a node scored by B dirty; or no node state at all."""
    for name, nut in (('a', [dict(commit='6' * 40, dirty=True, luc_chay=LUC_CHAM)]), ('b', [])):
        d, goc = build_c8(tmp_path / name)
        x = d['o_c8'][0]['chuoi'][0]
        old = x['file_cham']['sha256']
        _rewrite(goc, x['file_cham'], lambda y: y['scorer'].update(nut=nut))
        _relink(goc, d, x, old)
        with pytest.raises(GC.GateFileError, match='GC10-2'):
            GC.check_nguon(d, goc)


def test_round5_edo_consistent_with_itself(tmp_path):
    """GC10-4 (Codex): vuot_moi_nut true while no node was scored."""
    d, goc = build_c8(tmp_path / 'a')
    x = d['o_c8'][0]['chuoi'][0]
    _rewrite(goc, x['file_edo'], lambda e: (e.update(vuot_moi_nut=True), [n.update(cham=False) for n in e['theo_nut']]))
    _relink(goc, d, x)
    with pytest.raises(GC.GateFileError, match='GC10-4'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'b')                                  # every node scored, none with a frame above D
    x = d['o_c8'][0]['chuoi'][0]
    _rewrite(goc, x['file_edo'], lambda e: e.update(vuot_moi_nut=True))
    _relink(goc, d, x)
    with pytest.raises(GC.GateFileError, match='without a frame above D'):
        GC.check_nguon(d, goc)


def test_round5_drift_vuot_do_thieu(tmp_path):
    """GC10-1 (TC3-10): a missing quantity with the other above its threshold: vuot and vuot_do_thieu both true."""
    eps = dict(mm=1.0, huong_deg=2.0)
    dr = dict(trang_thai_vi_tri='thieu', trang_thai_huong='so', D_H_huong_deg=3.0, vuot_do_thieu=True)
    assert GC._drift_vuot(dr, eps, 'x') is True
    with pytest.raises(GC.GateFileError, match='GC10-1'):
        GC._drift_vuot(dict(dr, vuot_do_thieu=False), eps, 'x')


def test_round5_ghep_log_of_a_lost_run_copies_and_out_folder(tmp_path):
    """GC6-1: ghep finds the log of a lost run without a time file by the name thu_tu_clip gives; GC9-7: a copy of
    luoi.json under another name does not hide the grid; GC9-8: --out in a missing folder leaves no dossier."""
    d, goc = build_c8(tmp_path / 'a')
    _add_lost(goc, d, 0, 1, ['he_do'], gio=False, d1='khong_kiem')
    x = d['o_c8'][0]['chuoi'][1]
    lu = os.path.join(goc, x['file_cham']['duong_dan'])
    with open(lu, 'rb') as f, open(os.path.join(goc, 'aaa_copy.json'), 'wb') as g_:
        g_.write(f.read())
    g = _ghep(goc)
    assert g['o_c8'][0]['chuoi'][0]['file_log'] is not None
    GC.write_gate(str(tmp_path / 'g.json'), g, goc)
    d, goc = build_c8(tmp_path / 'b')
    with pytest.raises(OSError):
        GC.ghep_va_ghi(str(tmp_path / 'b' / 'no_such' / 'g.json'), goc, 'ke_hoach_c8.json', 'thu_tu_clip.csv',
                       'c8b_khoa_v5.json', 'tien_de_c8.json', 'kiem_cheo.json', dict(BANG_CHUNG), 'chuan_u/ho_so_new.json')
    assert not os.path.exists(os.path.join(goc, 'chuan_u', 'ho_so_new.json'))


def _add_k(goc, d, run_id='C8-P-K1-1', luc=LUC_CHAM):
    """A K run of C8 (its grid only: GC9-2), written in thu_tu_clip.csv and the dossier."""
    cell = dict(muc='O1-' + run_id.split('-')[2], controller='PID')
    rid = run_id.replace('-', '_')
    fk, _ = _luoi(goc, 'K_' + rid, cell, None, 0.005, clip='IMG_' + rid,
                  script=dict(commit='c' * 40, dirty=False, luc_chay=luc))
    d['luot_k'].append(dict(run_id=run_id, clip='IMG_' + rid, file_cham=fk, file_cham_h1=None))
    _write_csv(goc, d)
    runs = [x['run_id'] for c in d['o_c8'] for x in c['chuoi']] + [x['run_id'] for x in d['luot_k']]
    cl = {r: LUC_CHAM for r in runs}
    cl[run_id] = luc
    _rewrite(goc, d['chuan_u']['ho_so_c8'], lambda y: y.update(giu_ngoai=runs, cham_luc=cl))
    return d['luot_k'][-1]


def test_round5_k_runs_held_out(tmp_path):
    """GC9-2 (CODE 20/11: every C8 run is held out, K runs too; C8B: one luoi.json per K run)."""
    d, goc = build_c8(tmp_path / 'a')
    _add_k(goc, d)
    GC.check_nguon(d, goc)
    g = _ghep(goc)                                                    # ghep finds it
    assert [x['run_id'] for x in g['luot_k']] == ['C8-P-K1-1'] and 'C8-P-K1-1' in json.load(open(
        os.path.join(goc, 'chuan_u/ho_so_ghep.json'), encoding='utf-8'))['giu_ngoai']
    GC.write_gate(str(tmp_path / 'g.json'), g, goc)
    d['luot_k'] = []                                                  # left out of the gate
    with pytest.raises(GC.GateFileError, match='GC9-2'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'b')                                 # Codex: a K run scored before the S7 commit
    _add_k(goc, d, luc='2026-11-21T08:30:00+07:00')
    with pytest.raises(GC.GateFileError, match='not before the first scoring of a held-out run'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'c')                                 # a K grid of another task
    x = _add_k(goc, d)
    _rewrite(goc, x['file_cham'], lambda y: y.update(task='O1-K2'))
    with pytest.raises(GC.GateFileError, match='task O1-K1'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'd')                                 # a K grid of another clip
    x = _add_k(goc, d)
    _rewrite(goc, x['file_cham'], lambda y: y['clip'].update(ten='IMG_other'))
    with pytest.raises(GC.GateFileError, match='task O1-K1 for clip'):
        GC.check_nguon(d, goc)


def _second_scoring(goc, ref, rel):
    """Another first scoring of the clip of ref (other bytes: outcome khong_dat) at rel under goc."""
    lu = json.load(open(os.path.join(goc, ref['duong_dan']), encoding='utf-8'))
    lu['outcome'] = 'khong_dat'
    return _jw(goc, rel, lu)


def test_round5_runs_of_the_plan_scored_once(tmp_path):
    """CODE 20/11 (C8B dropped c8b_u cong, 7b75fcb; its luot_c8 check C8BU4-R1-3 is the gate's): every run of the gate is
    a run of the plan, scored once; the files of its chain are those its scoring names (so its time files too)."""
    d, goc = build_c8(tmp_path / 'a')                                 # a mistyped run code in thu_tu_clip.csv
    p = os.path.join(goc, 'thu_tu_clip.csv')
    with open(p, 'a', encoding='utf-8', newline='\n') as f:
        f.write('IMG_typo,C8-P-F05-1_lai,,,,,\n')
    d['thu_tu_clip']['sha256'] = GC.sha256_file(p)
    with pytest.raises(GC.GateFileError, match='mistyped'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'b')                                 # a copy with the same bytes is the same scoring
    x = d['o_c8'][0]['chuoi'][0]
    os.makedirs(os.path.join(goc, 'ban_sao'))
    shutil.copyfile(os.path.join(goc, x['file_cham']['duong_dan']), os.path.join(goc, 'ban_sao', 'luoi.json'))
    GC.check_nguon(d, goc)
    _second_scoring(goc, x['file_cham'], 'cham_lai/luoi.json')         # a second first scoring of that clip
    with pytest.raises(GC.GateFileError, match=r"has 2 first scoring\(s\)"):
        GC.check_nguon(d, goc)
    with pytest.raises(GC.GateFileError, match='2 first scorings'):  # ghep refuses it too
        _ghep(goc)
    d, goc = build_c8(tmp_path / 'c')                                 # the same for a K run
    k = _add_k(goc, d)
    GC.check_nguon(d, goc)
    _second_scoring(goc, k['file_cham'], 'cham_lai_k/luoi.json')
    with pytest.raises(GC.GateFileError, match=r'K run C8-P-K1-1: .*has 2 first scoring\(s\)'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'd')                                 # the chain names another clip H comparison
    dr = d['o_c8'][0]['chuoi'][0]['file_so_h']
    y = json.load(open(os.path.join(goc, dr['duong_dan']), encoding='utf-8'))
    y['ghi_chu'] = 'ban khac'
    dr2 = _jw(goc, 'drift_khac.json', y)
    for c in d['o_c8']:
        for x in c['chuoi']:
            x['file_so_h'] = dict(dr2)
    with pytest.raises(GC.GateFileError, match='file_so_h is not the one its scoring'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'e', two=True)                       # a second time file its scoring does not name
    x = d['o_c8'][0]['chuoi'][0]
    lu = json.load(open(os.path.join(goc, x['file_cham']['duong_dan']), encoding='utf-8'))
    y = json.load(open(os.path.join(goc, x['file_gio_h1']['duong_dan']), encoding='utf-8'))
    y['ghi_chu'] = 'ban khac'
    _relink_nodes(goc, d, x, lu, file_gio_h1=_jw(goc, 'gio_h1_khac.json', y))
    with pytest.raises(GC.GateFileError, match='file_gio_h1 is not the one its scoring'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'f', two=True)                       # its second scoring no longer named: a 2nd first
    x = d['o_c8'][0]['chuoi'][0]
    lu = json.load(open(os.path.join(goc, x['file_cham']['duong_dan']), encoding='utf-8'))
    _relink_nodes(goc, d, x, lu, file_cham_h1=None, file_gio_h1=None)
    with pytest.raises(GC.GateFileError, match=r'has 2 first scoring\(s\)'):
        GC.check_nguon(d, goc)


def test_round3_log_hashed_and_named(tmp_path):
    """GC6-1 (round 3): the ESP32 log itself is hashed (not the SHA the time file states) and named as the clip log."""
    d, goc = build_c8(tmp_path / 'a')
    x = d['o_c8'][0]['chuoi'][0]
    with open(os.path.join(goc, x['file_log']['duong_dan']), 'ab') as f:
        f.write(b'x')
    with pytest.raises(GC.GateFileError, match='sha256 differs'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'b')
    x, y = d['o_c8'][0]['chuoi'][0], d['o_c8'][1]['chuoi'][0]
    x['file_log'] = dict(y['file_log'])                                 # another run's log
    with pytest.raises(GC.GateFileError, match='file_log'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'c')                                  # a log of the right name, other bytes
    x = d['o_c8'][0]['chuoi'][0]
    p = os.path.join(goc, 'log2', os.path.basename(x['file_log']['duong_dan']))
    os.makedirs(os.path.dirname(p))
    with open(p, 'wb') as f:
        f.write(b'another log')
    x['file_log'] = dict(duong_dan='log2/' + os.path.basename(p), sha256=GC.sha256_file(p))
    with pytest.raises(GC.GateFileError, match='is not the ESP32 log the time file read'):
        GC.check_nguon(d, goc)


def test_round3_clip_of_the_scoring(tmp_path):
    d, goc = build_c8(tmp_path)
    x = d['o_c8'][0]['chuoi'][0]
    old = x['file_cham']['sha256']
    _rewrite(goc, x['file_cham'], lambda y: y['clip'].update(ten='IMG_other'))
    _relink(goc, d, x, old)
    with pytest.raises(GC.GateFileError, match='clip'):
        GC.check_nguon(d, goc)


def test_round3_fake_signs_refused_on_the_real_path(tmp_path):
    """GC6-14 (round 3): a FAKE content with c8_dir changed still carries FAKE marks: refused by check_nguon and by the
    rq3_log reader without allow_fake."""
    g = _good()
    g['c8_dir'] = 'C8'
    assert not GC.is_fake(g) and GC.has_fake_sign(g)
    with pytest.raises(GC.GateFileError, match='FAKE content has no sources'):
        GC.check_nguon(g, str(tmp_path))
    p = str(tmp_path / 'g.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(g, f)
    with pytest.raises(rl.LogError, match='FAKE'):
        rl.read_gate_file(p)
    with pytest.raises(rl.LogError, match='FAKE'):
        rl.read_gate_file(p, allow_fake=True)                             # not a whole FAKE content either


def test_gc6_4_videos_of_the_premise(tmp_path):
    d, goc = build_c8(tmp_path)
    names = ('H_a', 'H_b', 'H_m', 'H_z')
    vids = {n: json.load(open(os.path.join(goc, f'clip_h/{n}.json'), encoding='utf-8'))['video_sha256'] for n in names}
    _rewrite(goc, d['tien_de_c8']['file'], lambda x: x['sha256'].update(clip_h=dict(vids, H_b='d' * 64)))
    with pytest.raises(GC.GateFileError, match='video of H_b'):
        GC.check_nguon(d, goc)
    _rewrite(goc, d['tien_de_c8']['file'], lambda x: x['sha256'].update(clip_h=vids))
    GC.check_nguon(d, goc)


def test_gc6_16_write_gate_on_a_boundary_case_with_its_sources(tmp_path, monkeypatch):
    """GC6-16 (round 2): a bien_nguong level through write_gate with the C8 folder: refused without Liam's choice,
    written with it."""
    real = GC.uhat_level

    def tight(level, blk, u_t, budget=GC.BUDGET):
        return real(level, blk, u_t, budget=GC.grid_points(u_t, -u_t, Schedule(FIG8[level]).Tcmd + u_t, GC.DELTA0_S))
    d, goc = build_c8(tmp_path, levels=('O1-F05',), u_t=0.02)
    monkeypatch.setattr(GC, 'uhat_level', tight)
    lv = 'O1-F05'
    hang = _straddle(lv)
    t = dict(a_m=0.1445, u_p=dict(hang_m=hang, he_so_r=0.0, u_phi_rad=0.0, hang_theo_u=[]),
             u_psi=dict(hang_rad=math.radians(0.1), hang_theo_u=[]))
    q = {'g3': dict(chon='chi_bao_cao', ngay='2026-11-20', phien='local_x', nguyen_van='FAKE')}
    d['dieu_2b']['per_level'][lv] = GC.level_block(lv, t['a_m'], t['u_p'], t['u_psi'], d['runs_c8'][lv], q)
    _rewrite_khoa(goc, d, lambda k: k['muc'][lv]['hang_dong'].update(hang_m=hang, hang_rad=math.radians(0.1)))
    with pytest.raises(GC.GateFileError, match='CAN LIAM QUYET'):
        GC.write_gate(str(tmp_path / 'a.json'), d, goc)
    assert not os.path.exists(str(tmp_path / 'a.json'))
    d['quyet_liam'] = {lv: q}
    GC.write_gate(str(tmp_path / 'b.json'), d, goc)
    assert 'g3' in GC.read_gate_c8(str(tmp_path / 'b.json')).report_only(lv)


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
