"""tools/gate_c8.py: the C8 gate file gate_c8_v2 (check, check_nguon, Uhat, mask, chains, evidence, premise).
CODE_GATE 19/11/2026, design CLAUDE_THESIS 03_CODE/docs/2026-11-19_Thiet_ke_rq3_tuong_doi_v6_C8.md section 12 (after
the scoring rules J v6; tests taken from the counter-examples of Codex, design v5_C8 section 9). Every number is
invented (FAKE)."""
import copy
import hashlib
import json
import math
import os
import re
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
_PRE = ['']           # CODE 22/11: the folder of the C8 session the helpers write into (build_c8 sub; '' the C8 folder)


@pytest.fixture(autouse=True)
def _no_session_folder():
    _PRE[0] = ''
    yield
    _PRE[0] = ''


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
    assert uh['trang_thai'] in ('quyet', 'so_do') and uh['delta_cuoi_s'] < GC.DELTA0_S
    assert uh['uhat_duoi'] <= uh['uhat']
    assert (uh['uhat'] <= 0.05) if uh['trang_thai'] == 'quyet' else (uh['uhat_duoi'] > 0.05)


def test_rt3_1_budget_spent_is_bien_nguong_and_needs_liam():
    hang = _straddle()
    blk = _blk('O1-F05', hang=hang, hang_psi_deg=0.1)                # G5 decides here (not report-only)
    n0 = GC.uhat_level('O1-F05', _blk('O1-F05', hang=0.0), 0.02)['g3']['so_lan_tinh']   # the first grid alone
    uh = GC.uhat_level('O1-F05', blk, 0.02, budget=n0 + 1)
    assert uh['g3']['trang_thai'] == 'bien_nguong'
    with pytest.raises(GC.GateFileError, match='CAN LIAM QUYET'):
        GC.so_do_of('O1-F05', uh)
    q = dict(g3=dict(chon='so_do', ngay='2026-11-20', phien='local_x', nguyen_van='FAKE'))
    assert GC.so_do_of('O1-F05', uh, q) == (('g3',), {'g3': ['quyet_liam']})
    q['g3']['chon'] = 'quyet'
    assert GC.so_do_of('O1-F05', uh, q) == ((), {})


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
    assert uh['g2']['trang_thai'] == 'so_do'


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
    assert GC.so_do_of('O1-F05', uh)[0] == ('g2', 'g3')


# ── mask (design section 5) ──────────────────────────────────────────────────────────────────────────────────

def test_g5_f08_always_so_do():
    d = GC.fake_gate(('O1-F08',), u_psi_deg=0.1, u_t_s=0.001)
    e = d['dieu_2b']['per_level']['O1-F08']
    assert e['uhat']['g5']['trang_thai'] == 'quyet' and e['so_do'] == ['g5']
    assert e['ly_do_so_do'] == {'g5': ['liam_g5_f08']}
    e['so_do'] = []
    with pytest.raises(GC.GateFileError, match='so_do'):
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

def test_premise_needs_the_clip_h_at_the_end_of_the_segment():
    d = _good()
    d['tien_de_c8'][0]['doan'][0]['cuoi'] = None
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
    ('O1-F08', 'SIM', 'script', 'B-chung', ()),
    ('O1-F08', 'HW', 'screen', 'A-PID', ('g5',)),
    ('O1-F05', 'HW', 'confirm', 'A-PID', ()),
    ('O1-K1', 'HW', 'final', 'A-PID', ()),
    ('O1-F08', 'HIL', 'hil_check', 'B-PID', ('g5',)),
    ('O1-F08', 'HIL', 'final_hil', 'RQ2', ()),
])
def test_so_do_by_run(tmp_path, level, platform, stage, campaign, want):
    gf = GC.read_gate_c8(_write(tmp_path, _good()))
    assert gf.run_so_do(level, platform, stage, campaign) == tuple(want)


def test_unknown_platform_refused(tmp_path):
    gf = GC.read_gate_c8(_write(tmp_path, _good()))
    with pytest.raises(GC.GateFileError, match='platform'):
        gf.run_so_do('O1-F08', 'CAR', 'final', 'A-PID')


# ── check_nguon on a C8 folder with real source files (design sections 3, 6-10) ──────────────────────────────

def _jw(goc, rel, obj):
    rel = _PRE[0] + rel
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
    p = os.path.join(goc, _PRE[0] + 'log', base + '.csv')
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'wb') as f:
        f.write(('esp32 ' + base).encode())
    return dict(duong_dan=f'{_PRE[0]}log/{base}.csv', sha256=GC.sha256_file(p))


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
             nodes_s=[0.0, 0.001], cua_so_ghep_log_s=[1.0, 4.0],
             script=dict(commit='5' * 40, dirty=False, luc_chay=LUC_CHAM))      # CODE 23/11: c8b_gio of commit_hil
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
            sc['gate_v3']['g5'].update(q=5.0, n_undef=0, chung=dict(q=5.0, n_undef=0))   # middle node q of a so_do G5
            if cell['muc'] == 'O1-F08':                                     # SD1-6: the node block of J's so_do
                sc['gate_v3'].update(version=GC.GATE_VERSION_SO_DO, so_do=['g5'], gate_file_sha256=GC.J_V6_SHA256)
                sc['gate_v3']['g5'].update(q_so_do=dict(nut=None, q=5.0, q_chung=5.0), nhanh='so_do')
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
             script=dict(commit='5' * 40, dirty=False, luc_chay=LUC_CHAM), nodes_ok=[not lost, not lost],
             scorer=None if lost else dict(commit='5' * 40, dirty=False,
                                           nut=[dict(commit='5' * 40, dirty=False, luc_chay=LUC_CHAM)] * 2),
             u_t_nodes=[None, None] if lost else [u_max, u_max], so_do=['g5'] if cell['muc'] == 'O1-F08' else [],
             states_ba_nhanh={'g5': 'dat'} if cell['muc'] == 'O1-F08' else {},
             q_so_do={'g5': dict(nut=0, q=5.0, q_chung=5.0)} if cell['muc'] == 'O1-F08' else {},
             nums=dict(g4=dict(hi=0.01, n_undef=0), g5=dict(q=5.0, n_undef=0, chung=dict(q=5.0))),
             states={c: 'dat' for c in o1_gate.CONDS},
             file_so_h=None, file_cham_h1=None, file_gio_h1=None,
             loai_hong=(['thao_tac'] if ops else ['he_do']) if lost else [])
    x.update(kw)
    return _jw(goc, f'luot/{rid}/luoi.json', x), mpp


def _write_csv(goc, d):
    """thu_tu_clip.csv from the chains in order (clip, buoc = cell, log_pi = the run's log) between the two clips H at
    the start of the camera segment and its clip H at the end (CODE 20/11 after C8 session 2)."""
    lines = ['clip,buoc,gio_pc_bam_quay,gio_pc_dung_quay,created_utc,log_pi,ghi_chu', 'IMG_9000,H_a,,,,,clip H dau',
             'IMG_9001,H_b,,,,,clip H dau']
    for c in d['o_c8']:
        for j, x in enumerate(c['chuoi']):
            b = c['o'] + (f'_lai{j}' if j else '')                    # a rerun row: <run code>_lai<n> (CODE 19/11)
            lines.append(f'{x["clip"]},{b},,,,{x["run_id"]}.csv,')
    for x in d.get('luot_k') or []:                                  # GC9-2: the K runs (no log for them here)
        lines.append(f'{x["clip"]},{x["run_id"]},,,,,')
    lines.append('IMG_9002,H_m,,,,,clip H cuoi doan')
    p = os.path.join(goc, _PRE[0] + 'thu_tu_clip.csv')
    with open(p, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    d['thu_tu_clip'] = [dict(duong_dan=_PRE[0] + 'thu_tu_clip.csv', sha256=GC.sha256_file(p))]


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


LUAT_LICH = dict(win_s=0.5, win_min_n=10, version='c7_moc_dich_v1')


def _lich(goc, ten, s0, sets, **kw):
    """CODE 22/11: the tap_moc.json and lich.json of a camera segment (lich made with its first clip H session s0);
    returns the manifest source {tap_moc, lich} naming them."""
    tm = _jw(goc, f'lich/{ten}/tap_moc.json', dict(xe=sets))
    li = dict(version=GC.LICH_VERSION, session_sha256=s0['sha256'], hoi_tu=True, tap_moc_file_sha256=tm['sha256'],
              luat=dict(LUAT_LICH))
    li.update(kw)
    return dict(tap_moc=tm['sha256'], lich=_jw(goc, f'lich/{ten}/lich.json', li))


def _man_nguon(goc, d, x, ng):
    """The U manifest of the scoring of run x takes the manifest source ng (its segment's tap_moc and lich)."""
    for m in d['camera']['c8b']['moc_dich']['nguon']['manifests']:
        if json.load(open(os.path.join(goc, m['duong_dan']), encoding='utf-8'))['file_cham_sha256'] == \
                x['file_cham']['sha256']:
            _rewrite(goc, m, lambda y: y['nguon'].update(ng))


def build_c8(tmp_path, levels=LEVELS, u_t=0.014, two=False, sub=''):
    """A C8 folder (FAKE numbers, real files, no FAKE mark) and a gate content that passes check and check_nguon; the
    session's files in its folder sub of the C8 folder (CODE 22/11; '' the C8 folder itself)."""
    goc = str(tmp_path / 'C8')
    _PRE[0] = sub + '/' if sub else ''
    os.makedirs(os.path.join(goc, sub))
    plan = [dict(o=f'{lv}_{c}', muc=lv, controller=c) for lv in levels for c in GC.CONTROLLERS]
    kh = _jw(goc, 'ke_hoach_c8.json', dict(format=GC.KE_HOACH_FORMAT, o=plan))
    d = GC.fake_gate(levels, u_t_s=u_t)
    names = ('H_a', 'H_b', 'H_m')                                      # one camera segment: dau1, dau2, cuoi
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
    md['nguon']['luat_lich'] = dict(version=GC.LICH_VERSION, luat=LUAT_LICH)     # CODE 22/11: the rule of lich
    ngm = dict({k: md['nguon'][k] for k in GC.MOC_NGUON}, **_lich(goc, 'doan2', clips[0], [[1, 2, 3]]))
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
    a6 = _jw(goc, 'c8b/a6/c8b_a6_v2.json', dict(version='c8b_a6_v2', tam_id='T1'))        # CODE 23/11: the A6 of the lock
    _jw(goc, 'c8b/xoay/xoay_v1.json', dict(schema='xoay_v1', a6_sha256=a6['sha256']))     # F4 made with it
    _jw(goc, 'nhac_tam/lan_lap.json', dict(schema='lan_lap_v1', a6=dict(sha256=a6['sha256'], tam_id='T1')))   # F8
    d['bo_cham']['phien_ban'] = 'tools/c8b_khoa_v5.py@' + 'c' * 40
    kv5 = _jw(goc, 'c8b_khoa_v5.json', dict(
        version=GC.KHOA_FORMAT, trang_thai='khoa', nhanh_tick=d['u_t_branch'], bang_man_trap=d['bo_cham']['bang_man_trap'],
        commit_hil='5' * 40,                                          # CODE 23/11: the scoring commit (bo_tick)
        muc_kha_thi=list(levels), mien_car_pose=d['camera']['c8b']['mien_car_pose'], eps_H=d['camera']['c8b']['eps_H'],
        khoa_truoc=dict(nguon=[dict(vai='ke_hoach_c8', file='ke_hoach_c8.json', sha256=kh['sha256']),
                               dict(vai='a6', file='a6.json', sha256=a6['sha256'])],
                        script=dict(file='tools/c8b_khoa_v5.py', commit='c' * 40, dirty=False), luc=LUC_KHOA_C8B,
                        commit_hil='5' * 40),
        muc={lv: dict(so_do=['g5'] if lv == 'O1-F08' else [],
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
    commit, when = _git_commit(goc, ks['duong_dan'], LUC_S7)
    kc = _jw(goc, 'kiem_cheo.json', dict(version=GC.KIEM_CHEO_VERSION, doan=dict(ten='doan2'),     # one per segment
                                         khoa_s7=dict(file=ks['duong_dan'], sha256=ks['sha256'], git_commit=commit,
                                                      git_time=when)))
    all_runs = [c['chuoi'][0]['run_id'] for c in cells]
    hs = _jw(goc, 'chuan_u/ho_so_c8.json', dict(format=GC.HO_SO_FORMAT, bang_chung={k: v['sha256'] for k, v in bc.items()},
                                                ke_hoach_c8_sha256=kh['sha256'], khoa_c8b_sha256=kv5['sha256'],
                                                giu_ngoai=all_runs, cham_luc={r: LUC_CHAM for r in all_runs},
                                                khoa_kiem_luc=[dict(buoi='C8_buoi2', kiem_cheo=kc, khoa_s7=ks,
                                                                    git_commit=commit, git_time=when)],
                                                cham_cu_bo=dict(luoi=[], liam=None)))
    td = _jw(goc, 'tien_de_c8.json', dict(format=GC.TIEN_DE_FORMAT, ket_qua='dat', session_id='C8_buoi2',
                                          doan=dict(ten='doan2'), vai_tro_clip_h=dict(zip(names, GC.TD_ROLES)),
                                          sha256=dict(sessions={n: c['sha256'] for n, c in zip(names, clips)},
                                                      kiem_cheo=kc['sha256'])))
    d.update(c8_dir='C8', written_by=dict(script='c8b_u', commit='c' * 40, dirty=False), ke_hoach_c8=kh, khoa_c8b=kv5,
             o_c8=cells, runs_c8=runs,
             tien_de_c8=[dict(buoi='C8_buoi2', doan=[dict(ten='doan2', dau=clips[:2], cuoi=clips[2], file=td,
                                                          kiem_cheo=kc)])])
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
    _rewrite(goc, d['tien_de_c8'][0]['doan'][0]['file'], lambda x: x.update(ket_qua='khong_dat'))
    with pytest.raises(GC.GateFileError, match='CAN LIAM QUYET'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'c')
    _rewrite(goc, d['tien_de_c8'][0]['doan'][0]['file'], lambda x: x['sha256']['sessions'].pop('H_m'))
    with pytest.raises(GC.GateFileError, match='gives the roles'):
        GC.check_nguon(d, goc)


def test_gc6_4_premise_by_role_and_session(tmp_path):
    """GC6-4: the clips H of a segment compared role by role (dau1, dau2, cuoi), of this session, readable."""
    d, goc = build_c8(tmp_path / 'a')
    ch = d['tien_de_c8'][0]['doan'][0]
    ch['dau'][0], ch['cuoi'] = ch['cuoi'], ch['dau'][0]                # same bytes, roles swapped
    with pytest.raises(GC.GateFileError, match='role dau1|session order|camera segment'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'b')
    d['tien_de_c8'][0]['doan'][0]['ten'] = 'doan1'
    with pytest.raises(GC.GateFileError, match='premise of session'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'c')
    _rewrite(goc, d['tien_de_c8'][0]['doan'][0]['dau'][1], lambda x: x.pop('fit_ids'))
    _rewrite(goc, d['tien_de_c8'][0]['doan'][0]['file'],
             lambda x: x['sha256']['sessions'].update(H_b=d['tien_de_c8'][0]['doan'][0]['dau'][1]['sha256']))
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
    kk = json.load(open(os.path.join(goc, d['chuan_u']['ho_so_c8']['duong_dan']), encoding='utf-8'))['khoa_kiem_luc'][0]
    kc = dict(kk['kiem_cheo'])
    _rewrite(goc, kc, lambda y: y['khoa_s7'].update(git_commit=c, git_time=t))
    _rewrite(goc, d['chuan_u']['ho_so_c8'], lambda y: y['khoa_kiem_luc'][0].update(kiem_cheo=kc, git_commit=c, git_time=t))


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
    kk = json.load(open(os.path.join(goc, d['chuan_u']['ho_so_c8']['duong_dan']), encoding='utf-8'))['khoa_kiem_luc'][0]
    ks, kc = dict(kk['khoa_s7']), dict(kk['kiem_cheo'])
    _rewrite(goc, ks, lambda y: y.update(hang_so_s7={'n': 1}))
    _rewrite(goc, kc, lambda y: y['khoa_s7'].update(sha256=ks['sha256']))
    _rewrite(goc, d['chuan_u']['ho_so_c8'], lambda y: y['khoa_kiem_luc'][0].update(kiem_cheo=kc, khoa_s7=ks))
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
    p = os.path.join(goc, _PRE[0] + 'thu_tu_clip.csv')
    lines = open(p, encoding='utf-8').read().splitlines()
    lines = edit(lines)
    with open(p, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    d['thu_tu_clip'][0]['sha256'] = GC.sha256_file(p)


def test_gc6_1_chain_against_the_clip_log(tmp_path):
    d, goc = build_c8(tmp_path / 'a')
    _add_lost(goc, d, 0, 1, ['he_do'], luoi_kw=dict(loai_hong=['he_do']))
    GC.check_nguon(d, goc)                                             # a lost run then its rerun: passes
    _csv_edit(goc, d, lambda ls: ls[:3] + [ls[4], ls[3]] + ls[5:])      # rerun logged before the lost run
    with pytest.raises(GC.GateFileError, match='in that order|GC9-4'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'b')                                  # a physical run left out of the chain
    _csv_edit(goc, d, lambda ls: ls[:3] + [f'IMG_X,{d["o_c8"][0]["o"]},,,,X.csv,'] + ls[3:])
    with pytest.raises(GC.GateFileError, match='none left out|GC9-4'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'c')                                  # a figure-8 run outside the plan
    _csv_edit(goc, d, lambda ls: ls + ['IMG_Y,C8-P-F05-3,,,,Y.csv,'])
    with pytest.raises(GC.GateFileError, match='no cell of the plan'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'd')                                  # the time file reads another log
    _csv_edit(goc, d, lambda ls: ls[:3] + [ls[3].replace('.csv,', '_other.csv,')] + ls[4:])
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
            ('g', lambda k: k['muc']['O1-F08'].update(so_do=[]), 'F08')]:
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
    with pytest.raises(GC.GateFileError, match='one per segment'):
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
    assert GC.so_do_of('O1-F08', uh) == (('g5',), {'g5': ['liam_g5_f08']})
    with pytest.raises(GC.GateFileError, match='CAN LIAM QUYET'):          # elsewhere bien_nguong needs Liam
        GC.so_do_of('O1-F05', uh)
    with pytest.raises(GC.GateFileError, match='not needed'):
        GC.so_do_of('O1-F08', uh, {'g5': dict(chon='quyet')})
    gf = GC.read_gate_c8(_write(tmp_path, GC.fake_gate(('O1-F05',))))      # F08 not a gate level
    assert gf.run_so_do('O1-F08', 'HW', 'final', 'A-PID') == ('g5',)
    assert gf.run_so_do('O1-F08', 'SIM', 'script', 'B-chung') == ()
    assert gf.run_so_do('O1-K1', 'HW', 'final', 'A-PID') == ()


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
    d['dieu_2b']['per_level']['O1-F08']['ly_do_so_do'] = {'g5': ['vuot_nua_nguong']}
    with pytest.raises(GC.GateFileError, match='ly_do_so_do'):
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
    q = {'g3': dict(chon='so_do', ngay='2026-11-20', phien='local_x', nguyen_van='FAKE')}
    d['dieu_2b']['per_level'][lv] = GC.level_block(lv, t['a_m'], t['u_p'], t['u_psi'], runs, q)
    with pytest.raises(GC.GateFileError, match='CAN LIAM QUYET'):
        GC.write_gate(str(tmp_path / 'a.json'), d)
    assert not os.path.exists(str(tmp_path / 'a.json'))
    d['quyet_liam'] = {lv: q}
    GC.write_gate(str(tmp_path / 'b.json'), d)
    assert 'g3' in GC.read_gate_c8(str(tmp_path / 'b.json')).so_do(lv)


# ── Codex round 2 of code v6_C8 (GC6-1..6, 10, GC7-1, 2, 5; Liam at CODE "Sửa + vòng 3 ngay trên TK4") ──────────

def test_gc7_1_lower_bound_above_half_keeps_the_mask():
    """Counter-example of Codex round 2: F08, U_t 27 ms, hang_m 60 mm, a null band from 1.1686358585973085 m/s: the
    lower bound (60 mm) is above half the threshold, so G2, G3 report only whatever cells may still meet the band; the
    number is unknown (uhat null) and the reasons say so; no choice of Liam is asked."""
    b = _blk('O1-F08', hang=0.060, bands_p=[dict(u_lo=0.0, u_hi=1.1686358585973085, gia_tri=0.0),
                                            dict(u_lo=1.1686358585973085, u_hi=5.0, gia_tri=None)])
    uh = GC.uhat_level('O1-F08', b, 0.027)
    for c in ('g2', 'g3'):
        assert uh[c]['trang_thai'] == 'so_do' and uh[c]['uhat'] is None and uh[c]['uhat_duoi'] > 0.05
    ro, why = GC.so_do_of('O1-F08', uh)
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


def _hai_doan(goc, d):
    """C8 session 2 (CODE 20/11): a second camera segment (H_c, H_d at its start, H_z at its end) holding the LADRC
    runs, with its own premise file and clip H comparison; the PID runs stay in the first (H_a, H_b ... H_m)."""
    seg1 = d['tien_de_c8'][0]['doan'][0]
    td1 = json.load(open(os.path.join(goc, seg1['file']['duong_dan']), encoding='utf-8'))
    names = ('H_c', 'H_d', 'H_z')
    new = [_jw(goc, f'clip_h/{n}.json', dict(kind=GC.SESSION_KIND, ten=n, video=f'{n}_0{i}_IMG_900{i}.MOV',
                                             video_sha256=_h('video ' + n), **SESSION))
           for i, n in zip((4, 5, 6), names)]
    kc1 = json.load(open(os.path.join(goc, seg1['kiem_cheo']['duong_dan']), encoding='utf-8'))
    kc2 = _jw(goc, 'kiem_cheo_doan3.json', dict(kc1, doan=dict(ten='doan3')))   # its own result, the same S7 lock
    td2 = _jw(goc, 'tien_de_c8_doan3.json', dict(format=GC.TIEN_DE_FORMAT, ket_qua='dat', session_id=td1['session_id'],
                                                 doan=dict(ten='doan3'), vai_tro_clip_h=dict(zip(names, GC.TD_ROLES)),
                                                 sha256=dict(sessions={n: c['sha256'] for n, c in zip(names, new)},
                                                             kiem_cheo=kc2['sha256'])))
    d['tien_de_c8'][0]['doan'].append(dict(ten='doan3', dau=new[:2], cuoi=new[2], file=td2, kiem_cheo=kc2))
    dr2 = _jw(goc, 'drift_doan3.json', _drift(False, LEVELS, new))
    lines = ['clip,buoc,gio_pc_bam_quay,gio_pc_dung_quay,created_utc,log_pi,ghi_chu', 'IMG_9000,H_a,,,,,',
             'IMG_9001,H_b,,,,,']
    rows = {'PID': [], 'LADRC': []}
    for c in d['o_c8']:
        for j, x in enumerate(c['chuoi']):
            rows[c['controller']].append(f'{x["clip"]},{c["o"] + (f"_lai{j}" if j else "")},,,,{x["run_id"]}.csv,')
            if c['controller'] == 'LADRC':
                x['file_so_h'] = dict(dr2)
                lu = json.load(open(os.path.join(goc, x['file_cham']['duong_dan']), encoding='utf-8'))
                _relink_nodes(goc, d, x, lu, file_so_h=dict(dr2))
    ng3 = _lich(goc, 'doan3', new[0], [[1, 2], [3]])                 # CODE 22/11: its own mark set, the same rule
    for c in d['o_c8']:
        if c['controller'] == 'LADRC':
            for x in c['chuoi']:
                _man_nguon(goc, d, x, ng3)
    lines += rows['PID'] + ['IMG_9002,H_m,,,,,', 'IMG_9004,H_c,,,,,', 'IMG_9005,H_d,,,,,'] + rows['LADRC'] \
        + ['IMG_9006,H_z,,,,,']
    p = os.path.join(goc, 'thu_tu_clip.csv')
    with open(p, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    d['thu_tu_clip'] = [dict(duong_dan='thu_tu_clip.csv', sha256=GC.sha256_file(p))]


def test_gc6_3_the_drift_of_the_run_part(tmp_path):
    """CODE 20/11 after C8 session 2 (three camera segments): each run takes the clip H comparison of its segment (a
    clip H at its start against its clip H at the end), one premise file per segment."""
    d, goc = build_c8(tmp_path / 's1')
    _hai_doan(goc, d)
    GC.check_nguon(d, goc)                                            # PID in the first segment, LADRC in the second
    x = next(c for c in d['o_c8'] if c['controller'] == 'LADRC')['chuoi'][0]
    dr1 = dict(d['o_c8'][0]['chuoi'][0]['file_so_h'])                 # the comparison of the first segment
    lu = json.load(open(os.path.join(goc, x['file_cham']['duong_dan']), encoding='utf-8'))
    _relink_nodes(goc, d, x, lu, file_so_h=dr1)
    x['file_so_h'] = dr1
    with pytest.raises(GC.GateFileError, match='camera segment'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 's2')                                # the runs after the end of the only segment
    _csv_edit(goc, d, lambda ls: ls[:3] + [ls[-1]] + ls[3:-1])
    with pytest.raises(GC.GateFileError, match='lies in no camera segment'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 's3')                                # ghep: one premise file per segment, any order
    _hai_doan(goc, d)
    g = GC.ghep(goc, 'ke_hoach_c8.json', 'thu_tu_clip.csv', 'c8b_khoa_v5.json',
                ['tien_de_c8_doan3.json', 'tien_de_c8.json'], 'kiem_cheo.json', dict(BANG_CHUNG),
                'chuan_u/ho_so_ghep.json')
    assert [s['ten'] for s in g['tien_de_c8'][0]['doan']] == ['doan2', 'doan3'] and g['tien_de_c8'] == d['tien_de_c8']
    GC.write_gate(str(tmp_path / 'g.json'), g, goc)
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


def test_segment_kiem_cheo_and_the_comparison_of_the_segment(tmp_path):
    """CODE 20/11 after C8 session 2 (KIEMCHEO fix/c8-kiemcheo-doan: one kiem_cheo.json per segment): the kiem_cheo.json
    of each segment is of that segment and carries the S7 lock of the dossier; the dossier's is one of them; a clip H
    comparison of the segment compares a clip H at its start with its clip H at the end (not two clips at the start)."""
    def seg2(name):
        d, goc = build_c8(tmp_path / name)
        _hai_doan(goc, d)
        return d, goc, d['tien_de_c8'][0]['doan'][1]
    for name, edit in [('a', lambda y: y['doan'].update(ten='doan2')),                      # another segment's
                       ('b', lambda y: y['khoa_s7'].update(git_commit='e' * 40))]:           # another S7 lock
        d, goc, s = seg2(name)
        _rewrite(goc, s['kiem_cheo'], edit)
        _rewrite(goc, s['file'], lambda y: y['sha256'].update(kiem_cheo=s['kiem_cheo']['sha256']))
        with pytest.raises(GC.GateFileError, match='KCJ1-14'):
            GC.check_nguon(d, goc)
    d, goc, s = seg2('c')                                             # the dossier names a kiem_cheo of no segment
    old = d['tien_de_c8'][0]['doan'][0]['kiem_cheo']
    kc0 = json.load(open(os.path.join(goc, old['duong_dan']), encoding='utf-8'))
    d['tien_de_c8'][0]['doan'][0]['kiem_cheo'] = _jw(goc, 'kiem_cheo_doan2b.json', dict(kc0, ghi_chu='ban khac'))
    _rewrite(goc, d['tien_de_c8'][0]['doan'][0]['file'],
             lambda y: y['sha256'].update(kiem_cheo=d['tien_de_c8'][0]['doan'][0]['kiem_cheo']['sha256']))
    with pytest.raises(GC.GateFileError, match='kiem_cheo.json of no segment'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'd')                                 # a comparison of the two clips H at the start
    sh = [x['sha256'] for x in d['tien_de_c8'][0]['doan'][0]['dau']]
    dr = d['o_c8'][0]['chuoi'][0]['file_so_h']
    _rewrite(goc, dr, lambda y: y.update(s0_session_sha256=sh[0], s1_session_sha256=sh[1]))
    for c in d['o_c8']:
        x = c['chuoi'][0]
        x['file_so_h'] = dict(dr)
        lu = json.load(open(os.path.join(goc, x['file_cham']['duong_dan']), encoding='utf-8'))
        _relink_nodes(goc, d, x, lu, file_so_h=dict(dr))
    with pytest.raises(GC.GateFileError, match='camera segment'):
        GC.check_nguon(d, goc)
    d, goc, s = seg2('e')                                             # (H in the middle, H at the end): the old parts
    hm = d['tien_de_c8'][0]['doan'][0]['cuoi']['sha256']
    dr = dict(next(c for c in d['o_c8'] if c['controller'] == 'LADRC')['chuoi'][0]['file_so_h'])
    _rewrite(goc, dr, lambda y: y.update(s0_session_sha256=hm))
    for c in d['o_c8']:
        if c['controller'] == 'LADRC':
            x = c['chuoi'][0]
            x['file_so_h'] = dict(dr)
            lu = json.load(open(os.path.join(goc, x['file_cham']['duong_dan']), encoding='utf-8'))
            _relink_nodes(goc, d, x, lu, file_so_h=dict(dr))
    with pytest.raises(GC.GateFileError, match='camera segment'):
        GC.check_nguon(d, goc)
    d, goc, s = seg2('f')                                             # the segments listed out of session order
    d['tien_de_c8'][0]['doan'].reverse()
    with pytest.raises(GC.GateFileError, match='session order'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'g')                                 # its clip H at the end before those at the start
    _csv_edit(goc, d, lambda ls: [ls[0], ls[-1]] + ls[1:-1])
    with pytest.raises(GC.GateFileError, match='session order'):
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
    m = d['tien_de_c8'][0]['doan'][0]['dau'][1]
    _rewrite(goc, m, edit)
    _rewrite(goc, d['tien_de_c8'][0]['doan'][0]['file'], lambda x: x['sha256']['sessions'].update(H_b=m['sha256']))
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
    _rewrite(goc, d['camera']['c8b']['moc_dich']['nguon']['manifests'][0],
             lambda y: y['nguon'].update(bo_lay_mau='blm_v2'))
    with pytest.raises(GC.GateFileError, match='do not share one bo_lay_mau, car_pose'):
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
    _csv_edit(goc, d, lambda ls: ls[:3] + [ls[3].replace(',O1-F02_PID,', ',O1-F02_PID_lai2,')] + ls[4:])
    with pytest.raises(GC.GateFileError, match='GC9-4'):
        GC.check_nguon(d, goc)


def test_round4_grid_outcome_from_its_states(tmp_path):
    """GC9-5 (Codex): five states dat but outcome khong_dat is refused; a so_do condition decides too (CODE 23/11);
    a report_only list (before 23/11) is refused."""
    d, goc = build_c8(tmp_path / 'a')
    x = d['o_c8'][0]['chuoi'][0]
    old = x['file_cham']['sha256']
    _rewrite(goc, x['file_cham'], lambda y: y.update(outcome='khong_dat'))
    _relink(goc, d, x, old)
    with pytest.raises(GC.GateFileError, match='GC9-5'):
        GC.check_nguon(d, goc)
    fc = dict(trang_thai_luot='cham', outcome='khong_dat', so_do=['g5'],
              states=dict(g1='dat', g2='dat', g3='dat', g4='dat', g5='khong_dat'))
    GC._grid_outcome(fc, [], 'x')
    with pytest.raises(GC.GateFileError, match='GC9-5'):
        GC._grid_outcome(dict(fc, outcome='dat'), [], 'x')
    with pytest.raises(GC.GateFileError, match='report only is no rule'):
        GC._grid_outcome(dict(fc, report_only=['g5']), [], 'x')


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
        m = d['tien_de_c8'][0]['doan'][0]['dau'][1]
        _rewrite(goc, m, edit)
        _rewrite(goc, d['tien_de_c8'][0]['doan'][0]['file'], lambda x: x['sha256']['sessions'].update(H_b=m['sha256']))
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
    _rewrite(goc, d['tien_de_c8'][0]['doan'][0]['file'], lambda y: y['sha256'].update(kiem_cheo=kc_sha))
    g = _ghep(goc)
    assert g['o_c8'][0]['chuoi'][0]['file_edo'] == x['file_edo']
    GC.write_gate(str(tmp_path / 'g.json'), g, goc)
    d, goc = build_c8(tmp_path / 'b')
    with pytest.raises(GC.GateFileError, match='inside it'):
        _ghep(goc, ho_so='../outside.json')
    assert not os.path.exists(os.path.join(goc, '..', 'outside.json'))
    _rewrite(goc, d['tien_de_c8'][0]['doan'][0]['file'], lambda y: y.update(ket_qua='khong_dat'))     # the gate is refused
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
    s0 = d['tien_de_c8'][0]['doan'][0]['dau'][0]['sha256']
    _rewrite_nodes(goc, d, x, lambda y: y.update(session_sha256=s0), h1=True)       # the H0 calibration again
    with pytest.raises(GC.GateFileError, match='GC6-3'):
        GC.check_nguon(d, goc)


def test_round5_one_locked_model_and_the_premise_kiem_cheo(tmp_path):
    """GC6-4: the clip H sessions of one model (and the premise model); GC10-5: the premise names the kiem_cheo of
    the dossier."""
    d, goc = build_c8(tmp_path / 'a')
    m = d['tien_de_c8'][0]['doan'][0]['dau'][1]
    _rewrite(goc, m, lambda s: s.update(chosen='other', model_khoa='other'))
    _rewrite(goc, d['tien_de_c8'][0]['doan'][0]['file'], lambda x: x['sha256']['sessions'].update(H_b=m['sha256']))
    with pytest.raises(GC.GateFileError, match='GC6-3|GC6-4|part of the session'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'b')
    _rewrite(goc, d['tien_de_c8'][0]['doan'][0]['file'], lambda x: x.update(model='polynomial'))
    with pytest.raises(GC.GateFileError, match='GC6-4'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'c')
    _rewrite(goc, d['tien_de_c8'][0]['doan'][0]['file'], lambda x: x['sha256'].update(kiem_cheo='a' * 64))
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


def test_round5_grid_so_do_and_states_from_nodes(tmp_path):
    """GC9-5: so_do of a C8 grid is that of J (Codex: F05 with a mask g2 and outcome dat); a grid state dat needs
    every node dat; G1 by g1_rule."""
    for name, edit, msg in [
            ('a', lambda y: (y.update(so_do=['g2'], outcome='khong_dat'), y['states'].update(g2='khong_dat')), 'so_do'),
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
                  script=dict(commit='5' * 40, dirty=False, luc_chay=luc))
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
    d['thu_tu_clip'][0]['sha256'] = GC.sha256_file(p)
    with pytest.raises(GC.GateFileError, match='mistyped'):
        GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'a2')                                # session 2: the still clip before a K run
    p = os.path.join(goc, 'thu_tu_clip.csv')
    with open(p, 'a', encoding='utf-8', newline='\n') as f:
        f.write('IMG_3637,C8-P-K1-1_khoa,,,,,clip khoa K1\n')
    d['thu_tu_clip'][0]['sha256'] = GC.sha256_file(p)
    GC.check_nguon(d, goc)                                            # no run: neither a cell nor a K run
    with open(p, 'a', encoding='utf-8', newline='\n') as f:
        f.write('IMG_x,C8-P-F05-1_khoa,,,,,\n')                      # _khoa of a figure-8 code: no such clip
    d['thu_tu_clip'][0]['sha256'] = GC.sha256_file(p)
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
    names = ('H_a', 'H_b', 'H_m')
    vids = {n: json.load(open(os.path.join(goc, f'clip_h/{n}.json'), encoding='utf-8'))['video_sha256'] for n in names}
    _rewrite(goc, d['tien_de_c8'][0]['doan'][0]['file'], lambda x: x['sha256'].update(clip_h=dict(vids, H_b='d' * 64)))
    with pytest.raises(GC.GateFileError, match='video of H_b'):
        GC.check_nguon(d, goc)
    _rewrite(goc, d['tien_de_c8'][0]['doan'][0]['file'], lambda x: x['sha256'].update(clip_h=vids))
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
    q = {'g3': dict(chon='so_do', ngay='2026-11-20', phien='local_x', nguyen_van='FAKE')}
    d['dieu_2b']['per_level'][lv] = GC.level_block(lv, t['a_m'], t['u_p'], t['u_psi'], d['runs_c8'][lv], q)
    _rewrite_khoa(goc, d, lambda k: k['muc'][lv]['hang_dong'].update(hang_m=hang, hang_rad=math.radians(0.1)))
    with pytest.raises(GC.GateFileError, match='CAN LIAM QUYET'):
        GC.write_gate(str(tmp_path / 'a.json'), d, goc)
    assert not os.path.exists(str(tmp_path / 'a.json'))
    d['quyet_liam'] = {lv: q}
    GC.write_gate(str(tmp_path / 'b.json'), d, goc)
    assert 'g3' in GC.read_gate_c8(str(tmp_path / 'b.json')).so_do(lv)


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


# ── Liam 22/11 (C8 contract section 3 (a)): PID F08-1, F08-2 of session 2 recorded lost, not saved ───────────────

LIAM_HONG = dict(ngay='2026-11-22', phien='local_x', nguyen_van='FAKE ghi hong, khong cuu')


def _only_lost(goc, d, o, **kw):
    """Cell o with one run only, lost to the measurement (time file lost: ok false, hong_he_do), no rerun."""
    ci = next(i for i, c in enumerate(d['o_c8']) if c['o'] == o)
    gk = dict(ok=False, ket_luan='hong_he_do')
    gk.update(kw.pop('gio_kw', {}))
    _add_lost(goc, d, ci, 1, kw.pop('lh', ['he_do']), gio_kw=gk, **kw)
    d['o_c8'][ci]['chuoi'] = d['o_c8'][ci]['chuoi'][:1]
    _write_csv(goc, d)
    _fix_ho_so(goc, d)
    return ci


def test_liam_22_11_a_cell_kept_lost_without_a_rerun(tmp_path):
    o = 'O1-F08_PID'
    d, goc = build_c8(tmp_path / 'a')
    _only_lost(goc, d, o)
    with pytest.raises(GC.GateFileError, match='run it again'):         # without Liam's record: C8 is not done
        _ghep(goc)
    g = _ghep(goc, o_hong_theo_liam={o: LIAM_HONG})
    with pytest.raises(GC.GateFileError, match=r"O1-F08: no eligible run of \['PID'\]"):   # J v6: C8 not done there
        GC.write_gate(str(tmp_path / 'x.json'), g, goc)
    assert not os.path.exists(str(tmp_path / 'x.json'))
    g = _ghep(goc, o_hong_theo_liam={o: dict(LIAM_HONG)}, muc_bo_theo_liam={'O1-F08': dict(LIAM_HONG)})
    c = next(c for c in g['o_c8'] if c['o'] == o)
    assert c['trang_thai'] == 'hong_theo_liam' and c['liam'] == LIAM_HONG and c['chuoi'][-1]['ket_qua'] == 'hong'
    assert g['levels'] == ['O1-F02', 'O1-F05'] and [r['controller'] for r in g['runs_c8']['O1-F08']] == ['LADRC']
    p = str(tmp_path / 'gate.json')
    GC.write_gate(p, g, goc)                                            # check and check_nguon both pass
    assert GC.read_gate_c8(p).levels == ['O1-F02', 'O1-F05']
    # the gate file: the record only on a hong_theo_liam cell, and needed there
    for edit, msg in ((lambda c_: c_.pop('liam'), 'keys'),
                      (lambda c_: c_['liam'].pop('nguyen_van'), r'o_c8\[\d+\]\.liam'),
                      (lambda c_: c_.update(trang_thai='co_luot'), 'keys'),
                      (lambda c_: c_.update(trang_thai='co_luot') or c_.pop('liam'), 'run it again')):
        h = copy.deepcopy(g)
        edit(next(c_ for c_ in h['o_c8'] if c_['o'] == o))
        with pytest.raises(GC.GateFileError, match=msg):
            GC.check(h)
    # Liam's record on a cell whose last run is not lost, or a cell of no plan
    with pytest.raises(GC.GateFileError, match='only a run lost to the measurement'):
        _ghep(goc, o_hong_theo_liam={o: LIAM_HONG, 'O1-F05_PID': LIAM_HONG}, muc_bo_theo_liam={'O1-F08': LIAM_HONG})
    with pytest.raises(GC.GateFileError, match='cells of the plan'):
        _ghep(goc, o_hong_theo_liam={'O1-F11_PID': LIAM_HONG})
    with pytest.raises(GC.GateFileError, match='ngay'):
        _ghep(goc, o_hong_theo_liam={o: dict(LIAM_HONG, ngay='22/11')})
    # a cell closed at its cap (dong_tran) is no cell kept lost by Liam
    d, goc = build_c8(tmp_path / 'c')
    tb = dict(co_boot=False, t_boot_log_s=None, du_cua_so=False, trang_thai=None, thao_tac=[])
    ci = next(i for i, c_ in enumerate(d['o_c8']) if c_['o'] == o)
    for k in range(1, 5):
        _add_lost(goc, d, ci, k, ['he_do'], d1='truot', dem=True,
                  gio_kw=dict(ok=False, ket_luan='hong_he_do', dem_truot_kiem_gio=True,
                              kiem_gio_d1=dict(trang_thai='truot', truoc_boot=tb)))
    d['o_c8'][ci]['chuoi'] = d['o_c8'][ci]['chuoi'][:4]
    _write_csv(goc, d)
    _fix_ho_so(goc, d)
    assert next(c_ for c_ in _ghep(goc, ho_so='chuan_u/ho_so_c.json')['o_c8'] if c_['o'] == o)['trang_thai'] == 'dong_tran'
    with pytest.raises(GC.GateFileError, match="chain gives 'dong_tran'"):
        _ghep(goc, ho_so='chuan_u/ho_so_c2.json', o_hong_theo_liam={o: LIAM_HONG})
    # a run lost to an operation is run again whatever the record (J v6 section 5)
    d, goc = build_c8(tmp_path / 'b')
    _only_lost(goc, d, o, lh=['thao_tac', 'he_do'], ops=('boot',))
    with pytest.raises(GC.GateFileError, match='only a run lost to the measurement'):
        _ghep(goc, o_hong_theo_liam={o: LIAM_HONG}, muc_bo_theo_liam={'O1-F08': LIAM_HONG})


def test_liam_22_11_count_chain_and_cli_keys(tmp_path):
    assert GC.count_chain([_e('a', 'hong', ['he_do'])], hong_theo_liam=True)[0] == 'hong_theo_liam'
    for ch in ([_e('a')], [_e('a', 'hong', ['thao_tac', 'he_do'])], [_e('a', 'hong', ['cua_so_tham_chieu'])]):
        with pytest.raises(GC.GateFileError, match='only a run lost to the measurement'):
            GC.count_chain(ch, hong_theo_liam=True)
    assert GC.count_chain([_slip(c) for c in 'abcd'], hong_theo_liam=True)[0] == 'dong_tran'   # the cap first
    d, goc = build_c8(tmp_path)
    lj = str(tmp_path / 'liam.json')
    with open(lj, 'w', encoding='utf-8') as f:
        json.dump(dict(o_hong=LIAM_HONG), f)
    args = ['ghep', '--goc', goc, '--ke-hoach', 'ke_hoach_c8.json', '--thu-tu-clip', 'thu_tu_clip.csv',
            '--khoa-c8b', 'c8b_khoa_v5.json', '--tien-de', 'tien_de_c8.json', '--kiem-cheo', 'kiem_cheo.json',
            '--ho-so', 'chuan_u/ho_so_cli.json', '--out', str(tmp_path / 'gate.json'), '--liam', lj] + \
        sum((['--bang-chung', f'{k}={p}'] for k, p in BANG_CHUNG.items()), [])
    assert GC.main(args) == 1                                           # an unknown key of the --liam file
    with open(lj, 'w', encoding='utf-8') as f:
        json.dump(dict(o_hong_theo_liam={}), f)
    assert GC.main(args) == 0


# ── CODE 22/11 (C8 contract section 3; Liam at CODE "Chạy lại PID F08"): a rerun in a later floor session ──────────

def _buoi3(goc, d, o):
    """Cell o lost to the measurement in session b2 (no rerun there), rerun in a later floor session b3: its own folder,
    thu_tu_clip.csv, clips H, premise, kiem_cheo.json, clip H comparison and log. Returns the rerun."""
    ci = next(i for i, c in enumerate(d['o_c8']) if c['o'] == o)
    x = d['o_c8'][ci]['chuoi'][0]                                    # the valid run of the cell, now made in b3
    _only_lost(goc, d, o)
    _PRE[0] = 'b3/'
    names = ('H_e', 'H_f', 'H_y')
    clips = [_jw(goc, f'clip_h/{n}.json', dict(kind=GC.SESSION_KIND, ten=n, video=f'{n}_0{i}_IMG_910{i}.MOV',
                                               video_sha256=_h('video ' + n), **SESSION)) for i, n in enumerate(names)]
    kc0 = json.load(open(os.path.join(goc, d['tien_de_c8'][0]['doan'][0]['kiem_cheo']['duong_dan']), encoding='utf-8'))
    kc = _jw(goc, 'kiem_cheo.json', dict(kc0, doan=dict(ten='doan1')))   # its own result, the same S7 lock
    td = _jw(goc, 'tien_de_c8.json', dict(format=GC.TIEN_DE_FORMAT, ket_qua='dat', session_id='C8_buoi3',
                                          doan=dict(ten='doan1'), vai_tro_clip_h=dict(zip(names, GC.TD_ROLES)),
                                          sha256=dict(sessions={n: c['sha256'] for n, c in zip(names, clips)},
                                                      kiem_cheo=kc['sha256'])))
    dr = _jw(goc, 'drift.json', _drift(False, LEVELS, clips))
    lg = os.path.basename(x['file_log']['duong_dan'])
    os.makedirs(os.path.join(goc, 'b3', 'log'))
    os.replace(os.path.join(goc, x['file_log']['duong_dan']), os.path.join(goc, 'b3', 'log', lg))
    x['file_log'] = dict(x['file_log'], duong_dan='b3/log/' + lg)
    lu = json.load(open(os.path.join(goc, x['file_cham']['duong_dan']), encoding='utf-8'))
    _relink_nodes(goc, d, x, lu, file_so_h=dict(dr))
    x['file_so_h'] = dict(dr)
    _man_nguon(goc, d, x, _lich(goc, 'doan1', clips[0], [[2, 3]]))
    lines = ['clip,buoc,gio_pc_bam_quay,gio_pc_dung_quay,created_utc,log_pi,ghi_chu', 'IMG_9100,H_e,,,,,',
             'IMG_9101,H_f,,,,,', f'{x["clip"]},{o}_lai1,,,,pi/{lg},', 'IMG_9102,H_y,,,,,']
    p = os.path.join(goc, 'b3', 'thu_tu_clip.csv')
    with open(p, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    d['thu_tu_clip'].append(dict(duong_dan='b3/thu_tu_clip.csv', sha256=GC.sha256_file(p)))
    d['tien_de_c8'].append(dict(buoi='C8_buoi3', doan=[dict(ten='doan1', dau=clips[:2], cuoi=clips[2], file=td,
                                                            kiem_cheo=kc)]))
    d['o_c8'][ci]['chuoi'].append(x)
    _fix_ho_so(goc, d)
    hs0 = json.load(open(os.path.join(goc, d['chuan_u']['ho_so_c8']['duong_dan']), encoding='utf-8'))
    k0 = hs0['khoa_kiem_luc'][0]                     # CODE 24/11: the S7 lock of session b3 (the same manifest)
    _rewrite(goc, d['chuan_u']['ho_so_c8'], lambda y: y['khoa_kiem_luc'].append(dict(k0, buoi='C8_buoi3', kiem_cheo=kc)))
    _rewrite_khoa(goc, d, lambda k: k.update(mien_car_pose_theo_buoi={   # KG-5: the merged lock's domain per session
        b: {q: k['mien_car_pose'][q] for q in ('side_m', 'phien_ban', 'h_m', 'kappa')} for b in ('C8_buoi2', 'C8_buoi3')}))
    return x


def _ghep_hai_buoi(goc, tt=('b2/thu_tu_clip.csv', 'b3/thu_tu_clip.csv'),
                   td=('b3/tien_de_c8.json', 'b2/tien_de_c8.json'), ho_so='b2/chuan_u/ho_so_ghep.json'):
    kc = [t.split('/')[0] + '/kiem_cheo.json' for t in tt]          # CODE 24/11: one per session
    return GC.ghep(goc, 'b2/ke_hoach_c8.json', list(tt), 'b2/c8b_khoa_v5.json', list(td), kc,
                   {k: 'b2/' + v for k, v in BANG_CHUNG.items()}, ho_so)


def test_code_22_11_a_rerun_in_a_later_session(tmp_path):
    o = 'O1-F08_PID'
    d, goc = build_c8(tmp_path / 'a', sub='b2')                        # one session in its folder: as before
    x = _buoi3(goc, d, o)
    GC.check_nguon(d, goc)
    shutil.copyfile(os.path.join(goc, 'b3', 'kiem_cheo.json'), os.path.join(goc, 'b2', 'a_kc_b3.json'))   # same bytes
    g = _ghep_hai_buoi(goc)                                            # ghep takes the copy in the session's folder
    c = next(c_ for c_ in g['o_c8'] if c_['o'] == o)
    assert c['trang_thai'] == 'co_luot' and [e['run_id'] for e in c['chuoi']] == [o, o + '_lai1']
    assert [e['ket_qua'] for e in c['chuoi']] == ['hong', 'dat'] and c['chuoi'][1]['file_log'] == x['file_log']
    assert [t['buoi'] for t in g['tien_de_c8']] == ['C8_buoi2', 'C8_buoi3'] and g['tien_de_c8'] == d['tien_de_c8']
    assert g['thu_tu_clip'] == d['thu_tu_clip'] and g['levels'] == list(LEVELS)
    GC.write_gate(str(tmp_path / 'g.json'), g, goc)
    # without the later session the cell's last run is lost: C8 is not done
    with pytest.raises(GC.GateFileError, match='run it again'):
        _ghep_hai_buoi(goc, tt=['b2/thu_tu_clip.csv'], td=['b2/tien_de_c8.json'], ho_so='b2/chuan_u/ho_so_1.json')
    os.makedirs(os.path.join(goc, 'x'))                                # a premise in the folder of no session
    shutil.copyfile(os.path.join(goc, 'b3', 'tien_de_c8.json'), os.path.join(goc, 'x', 'tien_de_c8.json'))
    with pytest.raises(GC.GateFileError, match='folder of 0 C8 sessions'):
        _ghep_hai_buoi(goc, td=['b3/tien_de_c8.json', 'b2/tien_de_c8.json', 'x/tien_de_c8.json'],
                       ho_so='b2/chuan_u/ho_so_2.json')
    # the sessions in the wrong order: the rerun before its run
    with pytest.raises(GC.GateFileError, match='replaces a lost run only'):
        _ghep_hai_buoi(goc, tt=['b3/thu_tu_clip.csv', 'b2/thu_tu_clip.csv'], ho_so='b2/chuan_u/ho_so_3.json')
    # the log of the rerun in the folder of another session
    lg = os.path.basename(x['file_log']['duong_dan'])
    os.makedirs(os.path.join(goc, 'b2', 'log2'))
    shutil.copyfile(os.path.join(goc, x['file_log']['duong_dan']), os.path.join(goc, 'b2', 'log2', lg))
    h = copy.deepcopy(d)
    next(c_ for c_ in h['o_c8'] if c_['o'] == o)['chuoi'][1]['file_log']['duong_dan'] = 'b2/log2/' + lg
    with pytest.raises(GC.GateFileError, match='not in the folder'):
        GC.check_nguon(h, goc)
    os.replace(os.path.join(goc, x['file_log']['duong_dan']), os.path.join(goc, 'b3', 'log.bak'))
    with pytest.raises(GC.GateFileError, match='ESP32 log: no file'):     # ghep takes the log of its session only
        _ghep_hai_buoi(goc, ho_so='b2/chuan_u/ho_so_4.json')
    os.replace(os.path.join(goc, 'b3', 'log.bak'), os.path.join(goc, x['file_log']['duong_dan']))
    shutil.rmtree(os.path.join(goc, 'b2', 'log2'))
    # the gate file: one entry per session, distinct sessions and folders, each segment's files in its session folder
    for edit, msg in ((lambda h_: h_['tien_de_c8'].pop(), 'as many'),
                      (lambda h_: h_['tien_de_c8'][1].update(buoi='C8_buoi2'), 'each C8 session once'),
                      (lambda h_: h_['thu_tu_clip'][1].update(duong_dan='b2/x/thu_tu_clip.csv'), 'none inside another'),
                      (lambda h_: h_['thu_tu_clip'][1].update(duong_dan='b2/thu_tu_clip.csv'), 'none inside another'),
                      (lambda h_: h_['tien_de_c8'][1]['doan'][0].update(file=h_['tien_de_c8'][0]['doan'][0]['file']),
                       'not in the folder'),
                      (lambda h_: h_['tien_de_c8'][1]['doan'][0].update(ten='doan2'), None)):
        h = copy.deepcopy(g)
        edit(h)
        if msg is None:                                              # a segment name is unique within its session
            GC.check(h)
        else:
            with pytest.raises(GC.GateFileError, match=msg):
                GC.check(h)
    # the premise of the later session names the earlier session
    h = copy.deepcopy(d)
    _rewrite(goc, h['tien_de_c8'][1]['doan'][0]['file'], lambda y: y.update(session_id='C8_buoi2'))
    with pytest.raises(GC.GateFileError, match='the gate has it in session'):
        GC.check_nguon(h, goc)


def test_code_22_11_the_log_by_name_in_its_session(tmp_path):
    """A lost run without a time file: ghep finds its log by the name thu_tu_clip.csv gives, in its session's folder
    only (a log of that name in another session's folder is not it)."""
    d, goc = build_c8(tmp_path, sub='b2')
    lost = _add_lost(goc, d, 0, 1, ['he_do'], gio=False)
    _buoi3(goc, d, 'O1-F08_PID')
    lg = os.path.basename(lost['file_log']['duong_dan'])
    shutil.copyfile(os.path.join(goc, lost['file_log']['duong_dan']), os.path.join(goc, 'b3', 'log', lg))
    g = _ghep_hai_buoi(goc)
    assert g['o_c8'][0]['chuoi'][0]['file_log']['duong_dan'] == lost['file_log']['duong_dan']
    GC.write_gate(str(tmp_path / 'g.json'), g, goc)


def test_code_22_11_tap_moc_per_segment_one_lich_rule(tmp_path):
    """CODE 22/11 (after CAM): tap_moc is the result of lich per camera segment; the gate keeps one rule of lich for
    every segment and session, a tap_moc per segment, made with a session at the start of the run's segment."""
    d, goc = build_c8(tmp_path / 'a')
    _hai_doan(goc, d)                                     # two segments, two tap_moc, one rule: accepted
    GC.check_nguon(d, goc)
    seg2 = d['tien_de_c8'][0]['doan'][1]
    lad = [x for c in d['o_c8'] if c['controller'] == 'LADRC' for x in c['chuoi']]
    pid = [x for c in d['o_c8'] if c['controller'] == 'PID' for x in c['chuoi']]
    cases = [
        (lad, dict(luat=dict(LUAT_LICH, win_s=0.6)), 'its lich has the rule'),
        (lad, dict(version='c7_lich_v2'), 'converged'),
        (lad, dict(hoi_tu=False), 'converged'),
        (lad, dict(tap_moc_file_sha256='0' * 64), 'converged'),
        (lad[:1], dict(session_sha256=seg2['cuoi']['sha256']), 'no clip H at the start'),
        (pid[:1], dict(session_sha256=d['tien_de_c8'][0]['doan'][0]['dau'][0]['sha256']), 'another tap_moc'),
    ]
    mans = d['camera']['c8b']['moc_dich']['nguon']['manifests']
    orig = {m['duong_dan']: open(os.path.join(goc, m['duong_dan']), 'rb').read() for m in mans}
    for i, (xs, kw, msg) in enumerate(cases):
        h = copy.deepcopy(d)
        s0 = dict(sha256=kw.pop('session_sha256', seg2['dau'][0]['sha256']))
        ng = _lich(goc, f'x{i}', s0, [[9, i]], **kw)
        for x in xs:
            _man_nguon(goc, h, x, ng)
        with pytest.raises(GC.GateFileError, match=msg):
            GC.check_nguon(h, goc)
        for rel, raw in orig.items():                     # back to the segment's own lich
            with open(os.path.join(goc, rel), 'wb') as f:
                f.write(raw)
    GC.check_nguon(d, goc)
    for edit, msg in ((lambda y: y['nguon'].pop('lich'), 'nguon {bo_lay_mau, car_pose, tap_moc, lich}'),
                      (lambda y: y['nguon'].update(u_so='a' * 64), 'nguon {bo_lay_mau, car_pose, tap_moc, lich}'),
                      (lambda y: y['nguon'].update(bo_lay_mau='blm_v2'), 'bo_lay_mau, car_pose are not those')):
        h = copy.deepcopy(d)
        _rewrite(goc, h['camera']['c8b']['moc_dich']['nguon']['manifests'][0], edit)
        with pytest.raises(GC.GateFileError, match=msg):
            GC.check_nguon(h, goc)
        for rel, raw in orig.items():
            with open(os.path.join(goc, rel), 'wb') as f:
                f.write(raw)
    # ghep: one rule over every manifest
    d, goc = build_c8(tmp_path / 'b')
    _hai_doan(goc, d)
    for x in [x for c in d['o_c8'] if c['controller'] == 'LADRC' for x in c['chuoi']]:
        _man_nguon(goc, d, x, _lich(goc, 'doan3b', d['tien_de_c8'][0]['doan'][1]['dau'][0], [[1]],
                                    luat=dict(LUAT_LICH, win_s=0.6)))
    with pytest.raises(GC.GateFileError, match='lich files of 2 rules'):
        GC.ghep(goc, 'ke_hoach_c8.json', 'thu_tu_clip.csv', 'c8b_khoa_v5.json',
                ['tien_de_c8_doan3.json', 'tien_de_c8.json'], 'kiem_cheo.json', dict(BANG_CHUNG),
                'chuan_u/ho_so_ghep.json')
    g = copy.deepcopy(d)
    g['camera']['c8b']['moc_dich']['nguon']['luat_lich']['luat'] = {}
    with pytest.raises(GC.GateFileError, match='rule constants of lich'):
        GC.check(g)


# ── CODE 23/11: one A6 for the C8b lock and every F file (A6 sha 2aa8c970 -> d6bef847, same numbers) ─────────────

def test_code_23_11_one_a6_for_the_lock_and_the_f_files(tmp_path):
    d, goc = build_c8(tmp_path / 'a', sub='b2')                        # F4, F8 with the A6 of the lock: accepted
    old = _h('A6 cu')
    for rel, obj, field in (('b2/c8b/xoay_cu/xoay_v1.json', dict(schema='xoay_v1', a6_sha256=old), 'a6_sha256'),
                            ('b2/nhac_tam/lan_lap_cu.json', dict(schema='lan_lap_v1', a6=dict(sha256=old)), 'a6.sha256')):
        p = os.path.join(goc, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(obj, f)
        with pytest.raises(GC.GateFileError, match=re.escape(rel[3:]) + ' ' + re.escape(field)):
            GC.check_nguon(d, goc)
        q = os.path.join(goc, 'b2', '_archive', os.path.basename(os.path.dirname(rel)), os.path.basename(rel))
        os.makedirs(os.path.dirname(q), exist_ok=True)
        os.replace(p, q)                                               # superseded: kept in _archive, skipped
        GC.check_nguon(d, goc)
    # a later session's F file with another A6 is refused too
    d, goc = build_c8(tmp_path / 'b', sub='b2')
    _buoi3(goc, d, 'O1-F08_PID')
    GC.check_nguon(d, goc)
    with open(os.path.join(goc, 'b3', 'xoay_b3.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(schema='xoay_v1', a6_sha256=old), f)
    with pytest.raises(GC.GateFileError, match='b3/xoay_b3.json a6_sha256'):
        GC.check_nguon(d, goc)
    # the lock names no A6, or two
    for nguon_edit in (lambda n: [x for x in n if x['vai'] != 'a6'], lambda n: n + [dict(vai='a6', sha256=old)]):
        d, goc = build_c8(tmp_path / f'c{len(os.listdir(tmp_path))}')
        _rewrite_khoa(goc, d, lambda k: k['khoa_truoc'].update(nguon=nguon_edit(k['khoa_truoc']['nguon'])))
        with pytest.raises(GC.GateFileError, match='A6 file'):
            GC.check_nguon(d, goc)


# ── CODE 23/11 (chot T12 section 3, O1 Q4): bo_tick needs the scoring commit in the lock, every grid made by it ─────

def test_code_23_11_bo_tick_needs_the_scoring_commit_of_the_lock(tmp_path):
    d, goc = build_c8(tmp_path / 'a')                                  # bo_tick, commit_hil = the scorer: accepted
    assert d['u_t_branch'] == 'bo_tick'
    cases = ((lambda k: k.pop('commit_hil'), 'with commit_hil None'),
             (lambda k: k.update(commit_hil='5' * 39), 'with commit_hil'),
             (lambda k: k.update(commit_hil='6' * 40) or k['khoa_truoc'].update(commit_hil='6' * 40),
              'not at commit_hil 666666'),                           # KG-6: a scoring at another commit is dropped
             (lambda k: k['khoa_truoc'].update(commit_hil='6' * 40), 'khoa_truoc.commit_hil'))
    for i, (edit, msg) in enumerate(cases):
        d, goc = build_c8(tmp_path / f'b{i}')
        _rewrite_khoa(goc, d, edit)
        with pytest.raises(GC.GateFileError, match=msg):
            GC.check_nguon(d, goc)
    # the c8b_luoi and the final c8b_gio script of every scoring (CODE 23/11), clean
    x0 = None
    def other_scorer(y):                                               # the node scorer (c7_score) at another commit
        y['scorer'].update(commit='6' * 40)
        for e in y['scorer']['nut']:
            e.update(commit='6' * 40)
    for key, edit, msg in (('file_cham', lambda y: y['script'].update(commit='6' * 40), 'not at commit_hil'),   # KG-6
                           ('file_cham', other_scorer, 'is not commit_hil 555555'),
                           ('file_cham', lambda y: y['script'].update(dirty=True), 'luoi.json script'),
                           ('file_gio', lambda y: y['script'].update(commit='6' * 40), 'time file script'),
                           ('file_gio', lambda y: y.pop('script'), 'time file script')):
        d, goc = build_c8(tmp_path / f'e{len(os.listdir(tmp_path))}')
        x0 = d['o_c8'][0]['chuoi'][0]
        prev = x0['file_cham']['sha256']
        _rewrite(goc, x0[key], edit)
        _relink(goc, d, x0, old=prev)
        with pytest.raises(GC.GateFileError, match=msg):
            GC.check_nguon(d, goc)
    # a K run: its final time file found by the sha256 its luoi.json names, its script checked too
    d, goc = build_c8(tmp_path / 'k')
    _add_k(goc, d)
    GC.check_nguon(d, goc)
    raw = open(os.path.join(goc, d['luot_k'][0]['file_cham']['duong_dan']), 'rb').read()
    sha = d['luot_k'][0]['file_cham']['sha256']
    _rewrite(goc, d['luot_k'][0]['file_cham'], lambda y: y['script'].update(commit='6' * 40))
    with pytest.raises(GC.GateFileError, match='K run C8-P-K1-1: file_cham was made at commit .* not at commit_hil'):
        GC.check_nguon(d, goc)
    with open(os.path.join(goc, d['luot_k'][0]['file_cham']['duong_dan']), 'wb') as f:
        f.write(raw)
    d['luot_k'][0]['file_cham']['sha256'] = sha
    kcell = dict(muc='O1-K1', controller='PID')
    for commit, msg in (('6' * 40, 'K run time file'), ('5' * 40, None)):
        gk = _jw(goc, f'luot/K/gio_{commit[0]}.json', _gio('K_x', kcell, 0.005, script=dict(commit=commit, dirty=False)))
        _rewrite(goc, d['luot_k'][0]['file_cham'], lambda y: y.update(gio_sha256=gk['sha256']))
        if msg:
            with pytest.raises(GC.GateFileError, match=msg):
                GC.check_nguon(d, goc)
        else:
            GC.check_nguon(d, goc)
    os.remove(os.path.join(goc, gk['duong_dan']))
    with pytest.raises(GC.GateFileError, match='in no C8 session folder'):
        GC.check_nguon(d, goc)
    # giu_tick: the tick part kept, no commit asked
    d, goc = build_c8(tmp_path / 'c')
    _rewrite_khoa(goc, d, lambda k: k.update(nhanh_tick='giu_tick', commit_hil=None))
    d['u_t_branch'] = 'giu_tick'
    GC._check_commit_hil(d, goc, json.load(open(os.path.join(goc, d['khoa_c8b']['duong_dan']), encoding='utf-8')),
                         {'7' * 40})


# ── CODE 23/11 (03_CODE/docs/2026-11-23_Chot_luat_mat_na_quyet_bang_so_do_v1.md; Liam at CODE: "duyệt bảng luật mới"):
# a condition of the mask decides by its measured value q (branch so_do), no longer report only ──

def _gate_v32(theta=9.0, n=20, u_t=0.0015):
    frames = {'e_perp': [0.02] * n, 'e_trk': [0.03] * n, 'e_bao': [0.04] * n, 'e_theta_cmd_deg': [theta] * n}
    return rl.floor_gate(rl._plain(o1_gate.gate(frames, 'true', v_cmd_max=0.729, wz_cmd_max=1.634, u_t=u_t)))


def test_code_23_11_so_do_state_by_q():
    thr = 7.5
    assert GC.so_do_state(7.5, 7.5, 0, thr) == 'dat'                   # q at the threshold does not block
    assert GC.so_do_state(7.6, 7.6, 0, thr) == 'khong_dat'             # q above it blocks
    assert GC.so_do_state(7.0, 7.6, 0, thr) == 'chua_ket_luan'         # above it only in the union: not certain
    assert GC.so_do_state(7.0, 7.0, 1, thr) == 'chua_ket_luan'         # an undefined frame may lie above it
    assert GC.so_do_state(7.6, 7.6, 3, thr) == 'khong_dat'             # a measured value above it blocks anyway
    assert GC.so_do_state(float('nan'), float('nan'), 0, thr) == 'chua_ket_luan'
    assert GC.so_do_state(None, 7.0, 0, thr) == 'dat'                  # no common domain number: the union decides
    assert GC.so_do_state(7.0, 7.0, True, thr) == 'chua_ket_luan'      # a bool is no count


def test_code_23_11_apply_so_do_three_branch_kept_and_outcome_by_q():
    sha = 'a' * 64
    # theta 7.4 deg with U_t 15 ms: hi above 7.5 (three branches chua_ket_luan), q below it (so_do dat)
    g = _gate_v32(theta=7.4, u_t=0.015)
    assert g['g5']['state'] == 'chua_ket_luan' and g['outcome'] == 'chua_ket_luan'
    a = GC.apply_so_do(g, ['g5'], 'O1-F08', sha)
    assert a['version'] == GC.GATE_VERSION_SO_DO and a['so_do'] == ['g5'] and a['gate_file_sha256'] == sha
    assert a['g5']['state'] == 'dat' and a['g5']['state_ba_nhanh'] == 'chua_ket_luan' and a['g5']['nhanh'] == 'so_do'
    assert a['outcome'] == a['outcome_decide'] == 'dat' and a['complete'] is True and a['decide'] == list(o1_gate.CONDS)
    assert 'report_only' not in a and 'required' not in a and g['version'] == o1_gate.GATE_VERSION   # copy
    assert 'nhanh' not in a['g2'] and a['g2']['state'] == g['g2']['state']
    # 15 deg at F08 (Liam's example): khong_dat, it blocks the run
    b = GC.apply_so_do(_gate_v32(theta=15.0), ['g5'], 'O1-F08', sha)
    assert b['g5']['state'] == 'khong_dat' and b['outcome_decide'] == 'khong_dat' and b['any_certain_fail'] is True
    e = GC.apply_so_do(_gate_v32(theta=15.0), [], 'O1-F05', sha)                 # no so_do: three branches
    assert e['g5']['state'] == 'khong_dat' and e['so_do'] == [] and 'nhanh' not in e['g5']
    rl.validate_gate(a, 'HW', ('g5',))
    rl.validate_gate(b, 'HW', ('g5',))
    for bad, msg in [(dict(a, g5=dict(a['g5'], state='chua_ket_luan')), 'so_do'),
                     (dict(a, g5=dict(a['g5'], state_ba_nhanh='dat')), 'state_ba_nhanh'),
                     (dict(a, g5={k: v for k, v in a['g5'].items() if k != 'nhanh'}), 'nhanh'),
                     (dict(a, so_do=[]), 'gate file gives|nhanh'),
                     (dict(a, report_only=['g5']), 'report_only'),
                     (dict(a, gate_file_sha256=None), 'gate_file_sha256')]:
        with pytest.raises(rl.LogError, match=msg):
            rl.validate_gate(bad, 'HW', ('g5',))
    with pytest.raises(rl.LogError, match='gate file gives'):
        rl.validate_gate(a, 'HW', None)                                         # no gate file: no so_do
    with pytest.raises(rl.LogError, match='gate file gives'):
        rl.validate_gate(_gate_v32(), 'HW', ('g5',))                            # a three-branch gate where so_do is due


def test_code_23_11_apply_so_do_refuses():
    g = _gate_v32()
    for args, msg in [((g, ['g4'], 'O1-F08', 'a' * 64), 'so_do'), ((g, ['g5', 'g5'], 'O1-F08', 'a' * 64), 'so_do'),
                      ((g, ['g5'], '', 'a' * 64), 'level'), ((g, ['g5'], 'O1-F08', 'A' * 64), 'gate_file_sha256'),
                      ((dict(g, report_only=['g5']), ['g5'], 'O1-F08', 'a' * 64), 'report only is no rule'),
                      ((dict(g, version='gate_v3.4'), ['g5'], 'O1-F08', 'a' * 64), 'gate version')]:
        with pytest.raises(GC.GateFileError, match=msg):
            GC.apply_so_do(*args)


def test_code_23_11_grid_so_do_state_from_its_numbers(tmp_path):
    """A C8 grid at F08: G5 so_do, its state from nums (chung.q, q, n_undef), its three-branch state in
    states_ba_nhanh, the outcome over all five."""
    d, goc = build_c8(tmp_path / 'a')
    GC.check_nguon(d, goc)
    cell = next(c for c in d['o_c8'] if c['muc'] == 'O1-F08')
    for name, edit, msg in [
            ('b', lambda y: y['q_so_do']['g5'].update(q=8.0, q_chung=8.0), 'not the q_so_do .* of node 0'),
            ('f', lambda y: y['q_so_do']['g5'].update(nut=1), 'the node is 0'),
            ('g', lambda y: (y['states'].update(g5='chua_ket_luan'), y.update(outcome='chua_ket_luan')),
             'so_do from the middle node'),
            ('h', lambda y: y['nums']['g5'].update(n_undef=1), 'so_do from the middle node'),
            ('c', lambda y: y.pop('states_ba_nhanh'), 'states_ba_nhanh'),
            ('d', lambda y: y.update(so_do=[]), 'J says'),
            ('e', lambda y: y.update(report_only=['g5']), 'report only is no rule')]:
        d, goc = build_c8(tmp_path / name)
        x = next(c for c in d['o_c8'] if c['muc'] == 'O1-F08')['chuoi'][0]
        old = x['file_cham']['sha256']
        _rewrite(goc, x['file_cham'], edit)
        _relink(goc, d, x, old)
        with pytest.raises(GC.GateFileError, match=msg):
            GC.check_nguon(d, goc)
    assert cell['muc'] == 'O1-F08'


def test_code_24_11_ghep_writes_both_choices_of_a_boundary_case(tmp_path, monkeypatch):
    """CODE 24/11: a bien_nguong Uhat without Liam's choice (here G5 at F02): ghep stops, no gate file; the decision
    sheet beside --out holds both choices (inputs, Uhat, U_t,C8, half the threshold, the strict '>') and the so_do of
    the level each gives; with Liam's choice in --liam the gate is written with it."""
    real = GC.uhat_level

    def edge(level, blk, u_t, budget=GC.BUDGET):
        out = real(level, blk, u_t, budget)
        if level == 'O1-F02':
            out['g5'] = dict(uhat=None, uhat_duoi=3.6, delta_cuoi_s=GC.DELTA0_S, so_lan_tinh=7, trang_thai='bien_nguong')
        return out
    d, goc = build_c8(tmp_path)
    monkeypatch.setattr(GC, 'uhat_level', edge)                      # after the fixture: only ghep and its check
    out = str(tmp_path / 'gate.json')
    args = ['ghep', '--goc', goc, '--ke-hoach', 'ke_hoach_c8.json', '--thu-tu-clip', 'thu_tu_clip.csv',
            '--khoa-c8b', 'c8b_khoa_v5.json', '--tien-de', 'tien_de_c8.json', '--kiem-cheo', 'kiem_cheo.json',
            '--ho-so', 'chuan_u/ho_so_cli.json', '--out', out] + \
        sum((['--bang-chung', f'{k}={p}'] for k, p in BANG_CHUNG.items()), [])
    assert GC.main(args) == 3
    assert not os.path.exists(out) and not os.path.exists(os.path.join(goc, 'chuan_u/ho_so_cli.json'))
    ph = json.load(open(str(tmp_path / 'gate_phieu_chon.json'), encoding='utf-8'))
    assert ph['format'] == 'phieu_chon_v1' and len(ph['ca']) == 1
    ca = ph['ca'][0]
    assert (ca['muc'], ca['dieu_kien'], ca['nguong'], ca['nua_nguong']) == ('O1-F02', 'g5', 7.5, 3.75)
    assert ca['uhat']['trang_thai'] == 'bien_nguong' and ca['uhat']['uhat_duoi'] == 3.6 and 'chặt' in ca['so_sanh']
    assert ca['u_t_c8_s'] > 0 and ca['dau_vao']['luot'] and set(ca['dau_vao']) == {'a_m', 'u_p', 'u_psi', 'luot'}
    sd, q = ca['lua_chon']['so_do']['so_do_muc'], ca['lua_chon']['quyet']['so_do_muc']
    assert 'g5' in sd and 'g5' not in q and set(sd) - {'g5'} == set(q)
    assert GC.main(args) == 1                                         # the sheet is never replaced
    os.remove(str(tmp_path / 'gate_phieu_chon.json'))
    lp = str(tmp_path / 'liam.json')
    json.dump(dict(quyet_liam={'O1-F02': {'g5': dict(chon='so_do', ngay='2026-11-24', phien='local_x',
                                                     nguyen_van='FAKE')}}), open(lp, 'w', encoding='utf-8'))
    assert GC.main(args + ['--liam', lp]) == 0
    gf = GC.read_gate_c8(out)
    assert 'g5' in gf.so_do('O1-F02') and gf.data['dieu_2b']['per_level']['O1-F02']['ly_do_so_do']['g5'] == ['quyet_liam']


# ── Codex package 1 round 1 (SD1-n, 24/11; Liam at CODE_GATE "Sửa theo bảng (Recommended)") ──

def test_sd1_2_middle_node_tie_to_one_microsecond():
    import c8b_luoi as LU
    ns, ok = [511.9998, 512.0, 512.0002], [True, False, True]       # 0 and 2 the same distance from 512.0
    assert GC._nut_giua(ns, ok) == 0 == LU.nut_giua(ns, ok)
    assert GC._nut_giua([0.1, 0.3], [True, True]) == 0              # 0.2 - 0.1 vs 0.3 - 0.2 in float
    # round 2 (Codex): a tie on the edge of a rounding step (333.5 us each side on this c8b_gio lattice)
    import c8b_gio
    ns = c8b_gio.nodes(0.0012, 0.0010005)
    assert len(ns) == 4 and GC._nut_giua(ns, [True] * 4) == 1 == LU.nut_giua(ns, [True] * 4)
    assert GC._nut_giua(ns, [True, False, True, True]) == 2 == LU.nut_giua(ns, [True, False, True, True])
    assert GC.TIE_S == LU.TIE_S
    assert GC._nut_giua([0.0, 0.0010, 0.0011, 0.003], [True] * 4) == 2   # a real difference (0.1 ms) is no tie
    # CODE 24/11: the smallest real difference on a c8b_gio lattice is its spacing s > STEP_S / 2 (k = 2 with 2h just
    # above STEP_S gives s just above 0.5 ms): never a tie. Checked at a boot clock of 1000 s too
    for c in (0.0012, 1000.0):
        ns = c8b_gio.nodes(c, 0.5 * c8b_gio.STEP_S * (1 + 1e-6))
        assert len(ns) == 3 and 0.5 * c8b_gio.STEP_S < ns[1] - ns[0] < 0.51 * c8b_gio.STEP_S
        assert GC._nut_giua(ns, [True, True, False]) == 1 == LU.nut_giua(ns, [True, True, False])   # 0 is s farther
        assert GC._nut_giua(ns, [True, False, True]) == 0 == LU.nut_giua(ns, [True, False, True])   # a tie
    # the edge of TIE_S itself: 3 ns apart is no tie (the nearer node, the higher index), 0.5 ns apart is one
    assert GC._nut_giua([0.0, 0.399999997, 0.6, 1.0], [True] * 4) == 2 == LU.nut_giua([0.0, 0.399999997, 0.6, 1.0], [True] * 4)
    assert GC._nut_giua([0.0, 0.3999999995, 0.6, 1.0], [True] * 4) == 1


def test_sd3_1_grid_offsets_are_those_of_the_time_file_exactly():
    # Codex round 3: nodes(10.0, 0.0012); the middle node 1 moved by 5 ns in luoi.json passed the comparison with the
    # time file (a tolerance) and the gate then took node 2 as the middle one
    import c8b_gio
    ns = c8b_gio.nodes(10.0, 0.0012)
    assert len(ns) == 4 and GC._nut_giua(ns, [True] * 4) == 1
    moved = [ns[0], ns[1] - 5e-9, ns[2], ns[3]]
    assert GC._nut_giua(moved, [True] * 4) == 2                       # what the tolerance let through
    cell = dict(muc='O1-F08')
    with pytest.raises(GC.GateFileError, match='GC6-6'):
        GC._node_scores('.', dict(nodes_s=moved, trang_thai_luot='cham'), 'w', cell, dict(nodes_s=ns))
    with pytest.raises(GC.GateFileError, match='names the node scores'):    # the same offsets: past that check
        GC._node_scores('.', dict(nodes_s=list(ns), trang_thai_luot='cham'), 'w', cell, dict(nodes_s=ns))


def test_sd1_3_q_so_do_required_and_bounded():
    g = _gate_v32(theta=7.4, u_t=0.015)
    a = GC.apply_so_do(g, ['g5'], 'O1-F08', 'a' * 64)
    assert a['g5']['q_so_do'] == dict(nut=None, q=a['g5']['q'], q_chung=a['g5']['chung']['q'])
    for edit, msg in [(lambda x: x['g5'].pop('q_so_do'), 'q_so_do'),
                      (lambda x: x['g5']['q_so_do'].update(q_chung=7.45, q=7.42), 'q_chung'),
                      (lambda x: x['g5']['q_so_do'].update(q=9.0), 'above'),
                      (lambda x: x['g5']['q_so_do'].update(nut=2), 'nut'),
                      # round 2 (Codex): nut is null without a grid, nothing else; with no undefined frame q_so_do is
                      # the q of the domain (not null, not another number); no tolerance on the bounds (SD2-2)
                      (lambda x: x['g5']['q_so_do'].update(nut='x'), 'nut'),
                      (lambda x: (x['g5']['q_so_do'].update(q=None, q_chung=None),
                                  x['g5'].update(state='chua_ket_luan')), 'is undefined'),
                      (lambda x: (x['g5']['q_so_do'].update(q=7.3, q_chung=7.3)), 'is undefined'),
                      # round 3 (Codex): the count written 0.0 is still no undefined frame
                      (lambda x: (x['g5'].update(n_undef=0.0), x['g5']['chung'].update(n_undef=0.0),
                                  x['g5']['q_so_do'].update(q=7.3, q_chung=7.3)), 'is undefined'),
                      (lambda x: x['g5']['q_so_do'].update(q=x['g5']['q'] * (1 + 5e-10)), 'above')]:
        y = json.loads(json.dumps(a))
        edit(y)
        y.update(o1_gate.evaluate_outcomes(y, y['decide'], None))
        with pytest.raises(rl.LogError, match=msg):
            rl.validate_gate(y, 'HW', ('g5',))
    # the same on a logged row: the two q cells of the so_do condition emptied, the state turned chua_ket_luan
    cols = rl.gate_columns(a, 'a' * 64)
    back = rl.gate_from_row(dict(cols, run_uid='X', trajectory='O1-F08'))
    rl._recompute(back, 'HW', ('g5',))
    back = rl.gate_from_row(dict(cols, run_uid='X', trajectory='O1-F08', g5_so_do_q='', g5_so_do_q_chung='',
                                 g5_state='chua_ket_luan'))
    with pytest.raises(rl.LogError, match='is undefined'):
        rl._recompute(back, 'HW', ('g5',))
    # SD1-1: frames not all defined, no frame data (o1_gate block): q unknown, chua_ket_luan, never khong_dat
    u = json.loads(json.dumps(_gate_v32(theta=8.0)))
    u['g5']['n_undef'] = u['g5']['chung']['n_undef'] = 3
    b = GC.apply_so_do(u, ['g5'], 'O1-F08', 'a' * 64)
    assert b['g5']['q_so_do']['q'] is None and b['g5']['state'] == 'chua_ket_luan'


def test_sd1_4_every_node_with_the_exception_is_khong_dat():
    g = {'luoi': dict(cat_nut=[True, True, True]), 'decide': list(o1_gate.CONDS)}
    ev = dict(outcome_decide='chua_ket_luan', complete=True)
    assert rl.outcome_vs_gate('khong_dat', ev, g['decide'], rl.VIOL_BEFORE, g) is None
    assert rl.outcome_vs_gate('khong_dat', ev, g['decide'], '', g) == 'gate outcome chua_ket_luan'
    g['luoi']['cat_nut'] = [True, False, True]
    assert rl.outcome_vs_gate('khong_dat', ev, g['decide'], rl.VIOL_BEFORE, g) == 'gate outcome chua_ket_luan'


def test_sd1_6_every_scored_node_carries_the_so_do_of_j(tmp_path):
    for name, edit in [('a', lambda y: y['gate_v3'].update(gate_file_sha256='b' * 64)),
                       ('b', lambda y: y['gate_v3'].update(so_do=[])),
                       ('c', lambda y: y.update(report_only=['g5']))]:
        d, goc = build_c8(tmp_path / name)
        x = next(c for c in d['o_c8'] if c['muc'] == 'O1-F08')['chuoi'][0]
        _rewrite_nodes(goc, d, x, edit)
        with pytest.raises(GC.GateFileError, match='SD1-6'):
            GC.check_nguon(d, goc)
    d, goc = build_c8(tmp_path / 'd')                                  # a node of F05 with a so_do it has not
    x = next(c for c in d['o_c8'] if c['muc'] == 'O1-F05')['chuoi'][0]
    _rewrite_nodes(goc, d, x, lambda y: y['gate_v3'].update(so_do=['g5'], version=GC.GATE_VERSION_SO_DO,
                                                           gate_file_sha256=GC.J_V6_SHA256))
    with pytest.raises(GC.GateFileError, match='SD1-6'):
        GC.check_nguon(d, goc)


def test_sd1_7_a_row_with_the_old_column_is_refused():
    cols = rl.gate_columns(GC.apply_so_do(_gate_v32(), ['g5'], 'O1-F08', 'a' * 64), 'a' * 64)
    rl.gate_from_row(dict(cols, run_uid='X', trajectory='O1-F08'))
    with pytest.raises(rl.LogError, match='gate_report_only'):
        rl.gate_from_row(dict(cols, run_uid='X', trajectory='O1-F08', gate_report_only='g5'))
    # round 2 (Codex): also on a row without a gate (a lost run), where no gate is read back
    lost = dict(run_uid='L', run_outcome='hong_he_do', platform='HW', fail_cause='he_do')
    rl.check_gate_rows([dict(lost)])
    with pytest.raises(rl.LogError, match='gate_report_only'):
        rl.check_gate_rows([dict(lost, gate_report_only='g5')])


def test_sd2_1_node_q_so_do_tied_to_the_node_numbers(tmp_path):
    # Codex round 2: a node with no undefined frame and q 8 deg writing q_so_do 7.4 (and the grid copying it) was
    # accepted as dat: the node's q_so_do is its q when no frame is undefined, and its state follows it
    for name, edit, msg in [
            ('a', lambda y: y['gate_v3']['g5'].update(q=8.0, chung=dict(q=8.0, n_undef=0)), 'is undefined'),
            ('b', lambda y: y['gate_v3']['g5'].update(state='khong_dat'), 'SD2-1'),
            ('c', lambda y: y['gate_v3']['g5'].update(q_so_do=dict(nut=0, q=5.0, q_chung=5.0)), 'nut'),
            # round 3 (Codex): the grid copies the node's q_so_do, no tolerance (a node 1e-9 above, the grid below)
            ('d', lambda y: y['gate_v3']['g5'].update(q=5.000000001, chung=dict(q=5.0, n_undef=0),
                                                     q_so_do=dict(nut=None, q=5.000000001, q_chung=5.0)),
             'not the q_so_do'),                                       # q alone differs (2e-10 relative)
            ('d2', lambda y: y['gate_v3']['g5'].update(q=5.0, chung=dict(q=4.999999999, n_undef=0),
                                                      q_so_do=dict(nut=None, q=5.0, q_chung=4.999999999)),
             'not the q_so_do'),                                       # q_chung alone differs
            ('e', lambda y: y['gate_v3']['g5'].update(n_undef=0.0, q=8.0, chung=dict(q=8.0, n_undef=0.0)),
             'is undefined')]:
        d, goc = build_c8(tmp_path / name)
        x = next(c for c in d['o_c8'] if c['muc'] == 'O1-F08')['chuoi'][0]
        _rewrite_nodes(goc, d, x, edit)
        with pytest.raises(GC.GateFileError, match=msg):
            GC.check_nguon(d, goc)


def test_sd1_9_decision_sheet_with_two_open_cases_of_a_level():
    uh = {c: dict(uhat=None, uhat_duoi=0.04, delta_cuoi_s=GC.DELTA0_S, so_lan_tinh=1, trang_thai='bien_nguong')
          for c in ('g2', 'g5')}
    uh['g3'] = dict(uhat=0.03, uhat_duoi=0.02, delta_cuoi_s=GC.DELTA0_S, so_lan_tinh=1, trang_thai='quyet')
    base = dict(muc='O1-F02', uhat=uh, u_t=0.05, a_m=0.1, u_p={}, u_psi={}, runs=[], quyet_khac={})
    ph = GC.phieu_chon([dict(base, dieu_kien='g2', mo_khac=['g5']), dict(base, dieu_kien='g5', mo_khac=['g2'])])
    for ca in ph['ca']:
        assert ca['gia_dinh'] and ca['so_do_muc_moi_ca_bien_so_do'] == ['g2', 'g5']
        assert ca['lua_chon']['so_do']['so_do_muc'] == [ca['dieu_kien']] and ca['lua_chon']['quyet']['so_do_muc'] == []


def test_code_24_11_s7_lock_per_session(tmp_path):
    """CODE 24/11 (FABLE_C8 V2; GATE (d)): one khoa_s7 per C8 session in the dossier, its kiem_cheo of that session,
    the S7 numbers the same in every session, each lock before the first held-out scoring of its session."""
    o = 'O1-F08_PID'
    d, goc = build_c8(tmp_path / 'a', sub='b2')
    _buoi3(goc, d, o)
    GC.check_nguon(d, goc)
    _ghep_hai_buoi(goc)
    hs = json.load(open(os.path.join(goc, 'b2/chuan_u/ho_so_ghep.json'), encoding='utf-8'))
    assert hs['format'] == 'ho_so_c8_v2' and [k['buoi'] for k in hs['khoa_kiem_luc']] == ['C8_buoi2', 'C8_buoi3']
    assert hs['khoa_kiem_luc'][1]['kiem_cheo']['duong_dan'].startswith('b3/')
    bc = {k: 'b2/' + v for k, v in BANG_CHUNG.items()}
    tt, td = ['b2/thu_tu_clip.csv', 'b3/thu_tu_clip.csv'], ['b3/tien_de_c8.json', 'b2/tien_de_c8.json']
    with pytest.raises(GC.GateFileError, match='one per session'):        # one kiem_cheo for two sessions
        GC.ghep(goc, 'b2/ke_hoach_c8.json', tt, 'b2/c8b_khoa_v5.json', td, 'b2/kiem_cheo.json', bc,
                'b2/chuan_u/ho_so_x.json')
    with pytest.raises(GC.GateFileError, match='not in the folder of session'):   # a session's kiem_cheo elsewhere
        GC.ghep(goc, 'b2/ke_hoach_c8.json', tt, 'b2/c8b_khoa_v5.json', td, ['b2/kiem_cheo.json', 'b2/kiem_cheo.json'],
                bc, 'b2/chuan_u/ho_so_y.json')
    for edit, msg in [(lambda y: y['khoa_kiem_luc'][1].update(buoi='C8_buoi2'), 'session 1 is'),
                      (lambda y: y['khoa_kiem_luc'].pop(), 'one entry per C8 session')]:
        h = copy.deepcopy(d)
        _rewrite(goc, h['chuan_u']['ho_so_c8'], edit)
        with pytest.raises(GC.GateFileError, match=msg):
            GC.check_nguon(h, goc)
    # another S7 lock in session b3: other S7 numbers refused, the same numbers accepted
    for name, hang, ok in (('b', {'n': 2}, False), ('c', {}, True)):
        d, goc = build_c8(tmp_path / name, sub='b2')
        _buoi3(goc, d, o)
        ks = _jw(goc, 'khoa_s7_b3.json', dict(schema='c7_khoa_s7_v1', hang_so_s7=hang, commit_hil='e' * 40))
        c, t = _git_commit(goc, ks['duong_dan'], LUC_S7)
        kc = dict(d['tien_de_c8'][1]['doan'][0]['kiem_cheo'])
        _rewrite(goc, kc, lambda y: y['khoa_s7'].update(file=ks['duong_dan'], sha256=ks['sha256'], git_commit=c,
                                                         git_time=t))
        _rewrite(goc, d['tien_de_c8'][1]['doan'][0]['file'], lambda y: y['sha256'].update(kiem_cheo=kc['sha256']))
        d['tien_de_c8'][1]['doan'][0]['kiem_cheo'] = kc
        _rewrite(goc, d['chuan_u']['ho_so_c8'], lambda y: y['khoa_kiem_luc'][1].update(kiem_cheo=kc, khoa_s7=ks,
                                                                                    git_commit=c, git_time=t))
        if ok:
            GC.check_nguon(d, goc)
        else:
            with pytest.raises(GC.GateFileError, match='differ from those of session 0'):
                GC.check_nguon(d, goc)


# ── Codex N1 + merged lock round 1 (KG-1, KG-6; CODE 24/11) ──

def test_kg_1_camera_domain_with_its_state():
    d = _good()
    assert d['camera']['c8b']['mien_car_pose']['trang_thai'] == 'dat'
    GC.check(d)
    for edit, msg in [(lambda m: m.update(trang_thai='thieu', ly_do=['K2 thieu']), 'dat needed'),
                      (lambda m: m.pop('ly_do'), 'mien_car_pose')]:
        d = _good()
        edit(d['camera']['c8b']['mien_car_pose'])
        with pytest.raises(GC.GateFileError, match=msg):
            GC.check(d)


def test_kg_6_scorings_before_the_merged_lock_are_dropped(tmp_path):
    """A c8b_luoi file of a run made at another commit than commit_hil (a scoring before the N1 fix and the merged
    lock, kept in the folder, also under _archive): no scoring of the gate; the dossier lists it with Liam's record."""
    d, goc = build_c8(tmp_path / 'a')
    x = d['o_c8'][0]['chuoi'][0]
    old = json.load(open(os.path.join(goc, x['file_cham']['duong_dan']), encoding='utf-8'))
    old['script'] = dict(old['script'], commit='9' * 40)               # the same clip, scored before at another commit
    for rel in ('cu/luoi.json', '_archive/cu2/luoi.json'):
        _jw(goc, rel, old)
    with pytest.raises(GC.GateFileError, match='KG-6'):                # the dossier does not list them
        GC.check_nguon(d, goc)
    bo = GC.scorings_dropped(goc, '5' * 40)
    assert [e['duong_dan'] for e in bo] == sorted(['cu/luoi.json', '_archive/cu2/luoi.json'])
    rec = dict(ngay='2026-11-24', phien='local_42161302', nguyen_van='Một khóa, lấy lớn nhất')
    _rewrite(goc, d['chuan_u']['ho_so_c8'], lambda y: y.update(cham_cu_bo=dict(luoi=bo, liam=None)))
    with pytest.raises(GC.GateFileError, match='cham_cu_bo.liam'):     # listed without Liam's record
        GC.check_nguon(d, goc)
    _rewrite(goc, d['chuan_u']['ho_so_c8'], lambda y: y.update(cham_cu_bo=dict(luoi=bo, liam=rec)))
    GC.check_nguon(d, goc)                                             # the old scorings do not count as first ones
    with pytest.raises(GC.GateFileError, match='bo_cham_cu'):          # ghep needs Liam's record
        _ghep(goc, ho_so='chuan_u/ho_so_kg6a.json')
    g = _ghep(goc, ho_so='chuan_u/ho_so_kg6b.json', bo_cham_cu=rec)
    hs = json.load(open(os.path.join(goc, 'chuan_u/ho_so_kg6b.json'), encoding='utf-8'))
    assert hs['cham_cu_bo'] == dict(luoi=GC.scorings_dropped(goc, '5' * 40), liam=rec)
    assert g['o_c8'][0]['chuoi'][0]['file_cham']['sha256'] == x['file_cham']['sha256']


def test_kg_5_camera_domain_per_session(tmp_path):
    """KG-5: with two sessions the lock names each session's camera domain; the car pose of the manifests lies in the
    domain of each session with an eligible run, not only in the hull."""
    d, goc = build_c8(tmp_path / 'a', sub='b2')
    _buoi3(goc, d, 'O1-F08_PID')
    GC.check_nguon(d, goc)
    cp = d['camera']['c8b']['moc_dich']['nguon']['car_pose']
    k_c = cp['kappa'] if isinstance(cp, dict) else None
    for name, edit, msg in [
            ('b', lambda k: k.pop('mien_car_pose_theo_buoi'), 'KG-5'),
            ('c', lambda k: k['mien_car_pose_theo_buoi'].pop('C8_buoi3'), 'KG-5'),
            ('d', lambda k: k['mien_car_pose_theo_buoi']['C8_buoi3'].update(kappa=[k_c + 1.0, k_c + 2.0]),
             'outside the camera domain of session C8_buoi3')]:
        d, goc = build_c8(tmp_path / name, sub='b2')
        _buoi3(goc, d, 'O1-F08_PID')
        _rewrite_khoa(goc, d, edit)
        with pytest.raises(GC.GateFileError, match=msg):
            GC.check_nguon(d, goc)
