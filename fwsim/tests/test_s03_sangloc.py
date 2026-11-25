"""Tests of fwsim/s03_sangloc.py (spec S03 v8.6 section 3.1b): the committed numbers file, the
bounds check order, scoring of a gain set on the first-attempt eval rows, thresholds and boxes
on the nodes of W, the two independent checks, the sample size, the branches, the determinism
check, the sealed levels file, and the worker-process machinery. Fake runs are deterministic
functions of the gains; the real-process tests use child workers without the DLL except the
last one. Run: python -m pytest fwsim/tests/test_s03_sangloc.py -q"""
import hashlib
import io
import itertools
import json
import math
import os
import struct
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import fwsim                                     # noqa: E402
import s03_doe as s3                             # noqa: E402
import s04_ga as s4                              # noqa: E402
import s03_sangloc as sl                         # noqa: E402
import make_s03_screen_u as mk                   # noqa: E402

_DLL = os.path.join(fwsim.HERE, 'build', 'fwsim.dll')
needs_dll = pytest.mark.skipif(not os.path.isfile(_DLL), reason='fwsim.dll not built')

MU = [0.413, 0.436, 0.473]
MU_MID = MU[1]
FAKE_DEFAULTS = dict(PID=dict(kp_xy=1.5, ki_xy=0.05, kp_wz=1.0, ki_wz=0.05, pos_kp=6.0, pos_ki=3.0,
                              k_psi=1.0),
                     ADRC=dict(omega_c_xy=8.0, omega_o_xy=30.0, omega_c_wz=8.0, omega_o_wz=30.0,
                               pos_kp=6.0, pos_ki=3.0, k_psi=1.0))


def _bits(x):
    return struct.pack('<d', float(x)) if isinstance(x, float) else x


# ── the numbers file (item 3, section 8 condition 9) ───────────────────────────────

def test_screen_u_committed_file_bit_exact():
    """The committed s03_screen_u.csv is generate(0) bit for bit and passes parse_screen_u."""
    sets, info = sl.load_screen_u(sl.SCREEN_U, trial=True)
    want = mk.generate(0)
    for s in sl.SETS:
        assert sets[s].shape == want[s].shape == (sl.SET_ROWS[s], 7)
        assert all(struct.pack('<d', float(sets[s][j, k])) == struct.pack('<d', float(want[s][j, k]))
                   for j in range(len(want[s])) for k in range(7))
    with open(sl.SCREEN_U, encoding='utf-8') as f:
        again = sl.parse_screen_u(f.read())
    for s in sl.SETS:
        np.testing.assert_array_equal(again[s], sets[s])


def test_screen_u_generate_seeds_and_write(tmp_path):
    """generate(1) uses the seeds SEED_BASE + 10 + t and differs from generate(0); the tool-check
    base TRIAL_SEED_BASE gives seeds +5..+9 (never a main seed) and also differs."""
    one = mk.generate(1)
    for t, s in enumerate(sl.SETS):
        np.testing.assert_array_equal(
            one[s], np.random.default_rng(sl.SEED_BASE + 10 + t).random((sl.SET_ROWS[s], 7)))
        assert not np.array_equal(one[s], mk.generate(0)[s])
    thu = mk.generate(0, base=sl.TRIAL_SEED_BASE)
    for t, s in enumerate(sl.SETS):
        np.testing.assert_array_equal(
            thu[s], np.random.default_rng(sl.TRIAL_SEED_BASE + t).random((sl.SET_ROWS[s], 7)))
        assert sl.TRIAL_SEED_BASE + t not in {sl.SEED_BASE + 10 * r for r in range(3)}
    p = str(tmp_path / 'gen.csv')
    info = mk.write(p, r=1)                                 # a rerun file is allowed once
    assert os.path.isfile(p) and info['r'] == 1 and info['rows']['S0'] == sl.SET_ROWS['S0']
    back = sl.parse_screen_u(open(p, encoding='utf-8').read())
    np.testing.assert_array_equal(back['S0'], one['S0'])


def test_screen_u_write_refuses_existing(tmp_path):
    """The committed file is the reference: write() refuses an existing path and leaves no temp."""
    p = tmp_path / 'x.csv'
    p.write_text('touched', encoding='utf-8')
    with pytest.raises(Exception, match='exists'):
        mk.write(str(p))
    assert p.read_text(encoding='utf-8') == 'touched' and not (tmp_path / 'x.csv.tmp').exists()


def _u_text(head, rows):
    return '\n'.join([','.join(head)] + [','.join(r) for r in rows]) + '\n'


def test_parse_screen_u_failures():
    """condition 9: a wrong header, a set short or out of order, a row number not exactly str(j),
    a u not a finite float in [0, 1)."""
    head = sl.SCREEN_COLS
    u = ['0.5'] * 7
    with pytest.raises(s3.S03Stop, match='condition 9'):
        sl.parse_screen_u(_u_text(['set', 'row'] + [f'x{k}' for k in range(1, 8)],
                                  [['S0', '1'] + u]))
    with pytest.raises(s3.S03Stop, match='condition 9'):          # sets out of order
        sl.parse_screen_u(_u_text(head, [['S1', '1'] + u]))
    with pytest.raises(s3.S03Stop, match='condition 9'):          # row text is ' 1', not '1'
        sl.parse_screen_u(_u_text(head, [['S0', ' 1'] + u]))
    with pytest.raises(s3.S03Stop, match='condition 9'):          # numbering not continuous
        sl.parse_screen_u(_u_text(head, [['S0', '2'] + u]))
    for bad in ('1.0', '-1e-300', 'nan', 'inf', '-inf', 'x'):
        with pytest.raises(s3.S03Stop, match='condition 9'):
            sl.parse_screen_u(_u_text(head, [['S0', '1', bad] + u[1:]]))
    rows = [['S0', str(j)] + u for j in range(1, sl.SET_ROWS['S0'])] + [['S1', '1'] + u]
    with pytest.raises(s3.S03Stop, match='condition 9'):          # S0 one row short
        sl.parse_screen_u(_u_text(head, rows))


def test_load_screen_u_git_checks(tmp_path, monkeypatch):
    """load_screen_u reads the bytes once, records sha256/sha256_lf/sha256_crlf and the git state;
    a file not tracked/clean in git, or differing from its HEAD blob after CRLF->LF, is a
    S03Stop unless trial (then only recorded, .screen_info carries what was read)."""
    monkeypatch.setattr(sl, 'parse_screen_u', lambda t: {'S0': np.zeros((2, 7))})
    # C3-36: parse_screen_u is faked, so the source check is faked too (it would reject the
    # two-row arrays the fake returns)
    monkeypatch.setattr(sl, 'verify_screen_source',
                        lambda sets, sha256_lf, che_do='chinh', r=0: (dict(che_do=che_do, r=r), []))
    raw = b'a,b\r\n1,2\r\n'
    p = tmp_path / 'u.csv'
    p.write_bytes(raw)
    lf = raw.replace(b'\r\n', b'\n')
    monkeypatch.setattr(s3, 'file_git_state', lambda path: dict(tracked=True, clean=True))
    monkeypatch.setattr(s3, '_head_blob', lambda path: raw)
    sets, info = sl.load_screen_u(str(p))
    assert info['sha256'] == hashlib.sha256(raw).hexdigest()
    assert info['sha256_lf'] == hashlib.sha256(lf).hexdigest()
    assert info['sha256_crlf'] == hashlib.sha256(lf.replace(b'\n', b'\r\n')).hexdigest()
    assert info['matches_head_blob_lf'] is True and info['problems'] == []
    monkeypatch.setattr(s3, 'file_git_state', lambda path: dict(tracked=False, clean=False))
    with pytest.raises(s3.S03Stop) as exc:
        sl.load_screen_u(str(p))
    assert exc.value.screen_info['problems']
    sets, info = sl.load_screen_u(str(p), trial=True)
    assert info['problems'] and sets['S0'].shape == (2, 7)
    monkeypatch.setattr(s3, 'file_git_state', lambda path: dict(tracked=True, clean=True))
    monkeypatch.setattr(s3, '_head_blob', lambda path: b'different')
    with pytest.raises(s3.S03Stop, match='HEAD'):
        sl.load_screen_u(str(p))
    sets, info = sl.load_screen_u(str(p), trial=True)
    assert info['matches_head_blob_lf'] is False and info['problems']
    monkeypatch.setattr(sl, 'parse_screen_u',
                        lambda t: (_ for _ in ()).throw(s3.S03Stop('bad numbers')))
    with pytest.raises(s3.S03Stop):                             # parse failure stops even in trial
        sl.load_screen_u(str(p), trial=True)


# ── gains of a row and the per-set bounds check (item 4) ───────────────────────────

def _W():
    """The reference box of the fake steps: five geometric nodes per factor, ends inside the
    gain table (omega_o* ends at 48, kp_wz at 4.5 < 4.6, omega_c* at 150 < 200 so a high-u ADRC
    row can break the triangle)."""
    def g5(a, b):
        return list(np.geomspace(a, b, 5))
    common = dict(pos_kp=g5(1.0, 150.0), pos_ki=g5(1.0, 5000.0), k_psi=g5(0.5, 45.0))
    return {'PID': dict(common, kp_xy=g5(1.0, 32.0), ki_xy=g5(1.0, 1000.0), kp_wz=g5(0.5, 4.5),
                        ki_wz=g5(1.0, 200.0)),
            'ADRC': dict(common, omega_c_xy=g5(1.0, 150.0), omega_o_xy=g5(1.0, 48.0),
                         omega_c_wz=g5(1.0, 150.0), omega_o_wz=g5(1.0, 48.0))}


def _quet(W):
    return {c: {n: (ns[0], ns[-1]) for n, ns in W[c].items()} for c in W}


def test_gains_of_row_and_classify():
    """item 4: x_i = a_i (w_i/a_i)^u_i on the nodes of W; u = 0 gives a exactly, u just below 1
    stays <= w; classify_rows flags a triangle row ngoai_bang and an S03Stop carries
    controller/set/row."""
    W = _W()['PID']
    box = sl.w_box(W)
    assert box['kp_xy'] == (W['kp_xy'][0], W['kp_xy'][-1])
    g0 = sl.gains_of_row('PID', [0.0] * 7, W)
    assert all(g0[n] == W[n][0] for n in W)
    g9 = sl.gains_of_row('PID', [1.0 - 1e-12] * 7, W)
    assert all(W[n][0] <= g9[n] <= W[n][-1] for n in W)
    g5_ = sl.gains_of_row('PID', [0.5] * 7, W)
    assert g5_['kp_xy'] == W['kp_xy'][0] * (W['kp_xy'][-1] / W['kp_xy'][0]) ** 0.5
    recs = sl.classify_rows('PID', np.array([[0.5] * 7, [0.25] * 7]), W)
    assert [r['row'] for r in recs] == [1, 2] and all(r['flag'] == 'ok' for r in recs)
    recs = sl.classify_rows('ADRC', np.array([[0.99] * 7]), _W()['ADRC'])
    assert recs[0]['flag'] == 'ngoai_bang'                       # omega_c + 2 omega_o >= 200
    with pytest.raises(s3.S03Stop, match='PID S1 row 1'):
        sl.classify_rows('PID', np.array([[1.5] * 7]), W, 'S1')  # u > 1 -> x outside W


# ── item 5: scoring one gain set ────────────────────────────────────────────────────

def _erow(v, mu, e=0.05, tv=0.4, status='ok', attempt=1, completed='true', sha='s'):
    return dict(controller='PID', ga_seed='', gen='', cand_id=1, v=v, mu=mu, attempt=attempt,
                status=status, completed=completed, e_perp_max=e, tv_duty=tv, sat_frac=0.0,
                t_end=1.0, e_theta_lane_max_deg=0.5, e_theta_cmd_max_deg=0.5, ds_absmax=1.0,
                e_trk_max=0.05, e_bao_max=0.02, pen_e_perp_max=0, pen_e_trk_max=0,
                pen_e_bao_max=0, pen_e_theta_cmd_max_deg=0, log_sha256=sha, cause='', error='')


def _rows(mid=0.05, rest=0.05, g=1, err_mid=False, err_rest=False, step_b=True):
    speeds = [0.5, 0.8][:g]
    out = [_erow(v, MU_MID, e=mid, status='run_error' if err_mid else 'ok',
                completed='' if err_mid else 'true') for v in speeds]
    if step_b:
        out += [_erow(v, m, e=rest, status='run_error' if err_rest else 'ok',
                      completed='' if err_rest else 'true') for m in (MU[0], MU[2]) for v in speeds]
    return out


def test_feasible_and_mid_feasible():
    """feasible is exactly g1 == 0 and g2..g5 <= 0 of s4.score."""
    assert sl.feasible(dict(g1=0, g2=0.0, g3=-1e-9, g4=0.0, g5=0.0)) is True
    for k in ('g1', 'g2', 'g3', 'g4', 'g5'):
        s = dict(g1=0, g2=0.0, g3=0.0, g4=0.0, g5=0.0)
        s[k] = 1e-9
        assert sl.feasible(s) is False
    assert sl.mid_feasible(_rows(), MU_MID) is True
    assert sl.mid_feasible(_rows(mid=0.5), MU_MID) is False     # f1 > D


def test_score_modes_feasible_and_step_b():
    """item 5: a set feasible at the middle mu runs the two other levels (buoc_b); the ba_mu
    score is over all 3g runs."""
    rec = sl.score_modes(_rows(), 'ok', [0.5], MU)
    assert rec['giua']['kha_thi'] and rec['giua']['phan_loai_duoc'] and rec['giua']['n_runs'] == 1
    assert rec['ba_mu']['kha_thi'] and rec['ba_mu']['n_runs'] == 3 and rec['ba_mu']['buoc_b']
    rec = sl.score_modes(_rows(mid=0.05, rest=0.5), 'ok', [0.5], MU)
    assert rec['giua']['kha_thi'] and rec['ba_mu']['phan_loai_duoc'] and not rec['ba_mu']['kha_thi']
    rec = sl.score_modes(_rows(g=2), 'ok', [0.5, 0.8], MU)      # two gate speeds: 2 + 4 runs
    assert rec['giua']['n_runs'] == 2 and rec['ba_mu']['n_runs'] == 6


def test_score_modes_run_error():
    """item 5: a run_error run makes the set unclassifiable in the mode; not feasible at the
    middle mu -> never feasible at three mu and step b did not run (score fields None)."""
    rec = sl.score_modes(_rows(err_mid=True, step_b=False), 'ok', [0.5], MU)
    assert not rec['giua']['phan_loai_duoc'] and not rec['giua']['kha_thi']
    assert rec['ba_mu']['phan_loai_duoc'] is False and rec['ba_mu']['f1'] is None \
        and rec['ba_mu']['buoc_b'] is False
    rec = sl.score_modes(_rows(err_rest=True), 'ok', [0.5], MU)   # run_error only in step b
    assert rec['giua']['phan_loai_duoc'] and rec['giua']['kha_thi']
    assert not rec['ba_mu']['phan_loai_duoc'] and rec['ba_mu']['status'] == 'run_error'
    with pytest.raises(s3.S03Stop):                             # rest rows although not feasible
        sl.score_modes(_rows(err_mid=True, step_b=True), 'ok', [0.5], MU)
    with pytest.raises(s3.S03Stop):                             # a missing mid run is a tool error
        sl.score_modes([], 'ok', [0.5], MU)


def test_score_modes_tv_duty_and_ngoai_bang():
    """item 5: tv_duty missing or not finite arrives as run_error through s4.tv_duty_run_error
    (unclassifiable); ngoai_bang is classifiable, never feasible, scores g1 = g and 3g."""
    for bad in (float('nan'), '', None):
        r = s4.tv_duty_run_error(dict(status='ok', completed='true', tv_duty=bad))
        assert r['status'] == 'run_error'
    rec = sl.score_modes([dict(_erow(0.5, MU_MID), status='run_error')], 'ok', [0.5], MU)
    assert not rec['giua']['phan_loai_duoc']
    rec = sl.score_modes([], 'ngoai_bang', [0.5], MU)
    for mode, n in (('giua', 1), ('ba_mu', 3)):
        r = rec[mode]
        assert r['ngoai_bang'] and r['phan_loai_duoc'] and not r['kha_thi'] and not r['buoc_b']
        assert r['g1'] == n and r['f1'] == s4.PENALTY and r['status'] == 'ngoai_bang' \
            and r['n_runs'] == 0
    rec = sl.score_modes([], 'ngoai_bang', [0.5, 0.8], MU)        # g = 2: g1 = 2 and 6
    assert rec['giua']['g1'] == 2 and rec['ba_mu']['g1'] == 6


# ── items 6-8: thresholds, boxes on the nodes, the checks ──────────────────────────

def test_threshold_and_in_C():
    """item 6: tau = f1_min + F1_TOL over the feasible sets; f1 exactly tau is in C_j, the next
    float above is out; no feasible set -> no threshold."""
    recs = [dict(kha_thi=True, f1=0.2), dict(kha_thi=True, f1=0.05), dict(kha_thi=False, f1=0.01)]
    thr = sl.threshold(recs)
    assert thr['f1_min'] == 0.05 and thr['tau'] == 0.05 + s4.F1_TOL
    edge = thr['f1_min'] + s4.F1_TOL                            # the very expression of s4.select
    assert sl.in_C(dict(kha_thi=True, f1=edge), thr) is True
    assert sl.in_C(dict(kha_thi=True, f1=float(np.nextafter(edge, np.inf))), thr) is False
    assert sl.in_C(dict(kha_thi=False, f1=0.0), thr) is False
    assert sl.threshold([dict(kha_thi=False, f1=0.01)]) is None
    assert sl.in_C(dict(kha_thi=True, f1=0.0), None) is False


def test_nodes_and_boxes():
    """item 7: B_j is the smallest box on the nodes of W; a value exactly on a node widens by one
    node on each side that has a node; at the ends of W only one side moves."""
    nodes = [1.0, 2.0, 4.0, 8.0, 16.0]
    assert sl.node_floor(nodes, 3.0) == 2.0 and sl.node_ceil(nodes, 3.0) == 4.0
    assert sl.node_floor(nodes, 4.0) == 4.0 and sl.node_ceil(nodes, 4.0) == 4.0
    for x in (0.5, 17.0):
        with pytest.raises(s3.S03Stop):
            sl.node_floor(nodes, x)
        with pytest.raises(s3.S03Stop):
            sl.node_ceil(nodes, x)
    W = dict(f=nodes)
    assert sl.box_from_points([dict(f=3.0)], W) == dict(f=(2.0, 4.0))
    assert sl.box_from_points([dict(f=4.0)], W) == dict(f=(2.0, 8.0))      # on a node: widen both
    assert sl.box_from_points([dict(f=1.0)], W) == dict(f=(1.0, 2.0))      # first node: one side
    assert sl.box_from_points([dict(f=16.0)], W) == dict(f=(8.0, 16.0))    # last node: one side
    assert sl.box_from_points([dict(f=4.0 + 1e-9)], W) == dict(f=(4.0, 8.0))  # 1e-9 above a node
    assert sl.box_from_points([dict(f=3.0), dict(f=9.0)], W) == dict(f=(2.0, 16.0))
    with pytest.raises(s3.S03Stop):
        sl.box_from_points([], W)
    W_o = dict(o=list(np.geomspace(2.0, 48.0, 4)))
    assert sl.box_from_points([dict(o=48.0)], W_o)['o'][1] == 48.0         # omega_o ceiling
    assert sl.hull(dict(f=(2.0, 8.0)), dict(f=(4.0, 16.0))) == dict(f=(2.0, 16.0))
    assert sl.extend_box(dict(f=(4.0, 8.0)), [dict(f=9.0)], W) == dict(f=(4.0, 16.0))
    same = sl.extend_box(dict(f=(4.0, 8.0)), [dict(f=8.0)], W)             # NO widening rule
    assert same == dict(f=(4.0, 8.0))
    assert sl.extend_box(dict(f=(4.0, 8.0)), [], W) == dict(f=(4.0, 8.0))  # no points: same box
    assert sl.in_box(dict(f=8.0), dict(f=(4.0, 8.0))) is True              # an end is inside
    assert sl.in_box(dict(f=np.nextafter(8.0, np.inf)), dict(f=(4.0, 8.0))) is False


def test_check_counts_and_bounds():
    """item 8: m counts the C_j sets of the check sample, k those outside the box; accepted when
    k == 0 and m >= 86; bound 1 - 0.0125^(1/m) only when m >= 1 and k == 0."""
    thr = dict(f1_min=0.05, tau=0.06)
    recs = [dict(kha_thi=True, f1=0.05), dict(kha_thi=True, f1=0.07), dict(kha_thi=False, f1=0.0)]
    gains = [dict(f=5.0), dict(f=9.0), dict(f=1.0)]
    assert sl.check_counts(recs, gains, thr, dict(f=(4.0, 8.0))) == (1, 0)
    gains[0] = dict(f=9.0)
    assert sl.check_counts(recs, gains, thr, dict(f=(4.0, 8.0))) == (1, 1)
    assert sl.accepted(86, 0) is True and sl.accepted(85, 0) is False and sl.accepted(86, 1) is False
    assert sl.bound(86, 0) == pytest.approx(0.0497, abs=5e-4)
    assert sl.bound(100, 0) == pytest.approx(0.043, abs=1e-3)
    assert sl.bound(300, 0) == pytest.approx(0.0145, abs=5e-4)
    assert sl.bound(0, 0) is None and sl.bound(86, 1) is None


def test_over_error_limit_and_counts():
    """items 5, 10: strictly more than 10 % unclassifiable is over; ngoai_bang is classifiable
    (in the denominator, never the numerator); n_phan_loai_duoc includes it."""
    assert sl.over_error_limit(10, 100) is False                 # exactly 10 % is allowed
    assert sl.over_error_limit(11, 100) is True                  # one more is over
    recs = [dict(row=j, flag='ngoai_bang') for j in range(1, 10)] + [dict(row=10, flag='ok')]
    scores = {j: {m: dict(s4.score_ngoai_bang(1), phan_loai_duoc=True, kha_thi=False, buoc_b=False)
                  for m in sl.MODES} for j in range(1, 10)}
    scores[10] = {m: dict(phan_loai_duoc=False, kha_thi=False) for m in sl.MODES}
    st = sl._mode_stats(recs, scores)['giua']
    assert st['n_bo_gain'] == 10 and st['n_phan_loai_duoc'] == 9 and st['n_khong_phan_loai'] == 1
    assert sl.over_error_limit(st['n_khong_phan_loai'], st['n_bo_gain']) is False   # 1/10: not over


# ── item 10: the sample size ────────────────────────────────────────────────────────

def _q(counts, n0=1000):
    return {(c, m): counts.get((c, m), n0) for c in s3.CONTROLLERS for m in sl.MODES}


def test_sample_size_boundaries():
    """item 10: N_tk is the smallest multiple of 1000 in [3000, 10000] with N q_min >= 2 m_min
    (exact integers); no multiple -> 10000; N_gio the largest within 3 (N/1000) t_S0 <= 24 h."""
    sd = sl.sample_size(_q({('ADRC', 'ba_mu'): 43}), 1000, 60.0, False)
    assert sd['N_tk'] == 4000 and 4000 * 43 == 2 * 86 * 1000    # the exact boundary
    sd = sl.sample_size(_q({('ADRC', 'ba_mu'): 42}), 1000, 60.0, False)
    assert sd['N_tk'] == 5000                                    # one less crosses it
    sd = sl.sample_size(_q({('ADRC', 'ba_mu'): 17}), 1000, 60.0, False)
    assert sd['N_tk'] == 10000                                   # no multiple in the range
    t24 = sl.T_MAX_H * 3600.0 / (3 * 10)                         # exactly 24 h for N = 10000
    assert sl.sample_size(_q({}), 1000, t24, False)['N_gio'] == 10000
    assert sl.sample_size(_q({}), 1000, t24 + 1.0, False)['N_gio'] == 9000
    sd = sl.sample_size(_q({}), 1000, 28800.0 + 1.0, False)      # t_S0 > 8 h: nothing fits
    assert sd['N_gio'] == 0 and sd['N'] == 0 and sd['chay'] is False
    assert any('N = 0 < 3000' in r for r in sd['ly_do'])


def test_sample_size_t_s0_and_reasons():
    """item 10: t_S0 must be finite and positive; N * q_min < m_min or an over-10 % S0 -> no
    screening; q keeps the denominator n0; the reasons start with the branch name."""
    for t in (0.0, -1.0, float('nan'), float('inf')):
        with pytest.raises(s3.S03Stop):
            sl.sample_size(_q({}), 1000, t, False)
    sd = sl.sample_size(_q({('PID', 'giua'): 8, ('PID', 'ba_mu'): 8,
                            ('ADRC', 'giua'): 8, ('ADRC', 'ba_mu'): 8}), 1000, 60.0, False)
    assert sd['N_tk'] == 10000 and sd['N'] * sd['q_min'] < 86 and sd['chay'] is False
    assert any(r.startswith('khong_chay_sang_loc') and 'q_min' in r for r in sd['ly_do'])
    assert sd['q']['PID giua']['mau'] == 1000 and sd['q']['PID giua']['q'] == 8 / 1000
    sd = sl.sample_size(_q({}), 1000, 60.0, True)
    assert sd['chay'] is False and any('S0' in r for r in sd['ly_do'])
    assert sd['N_da_dung'] == 0


def test_sample_size_tool_check():
    """8.11: with tool_n the formula is computed and its two N-reasons recorded under
    ly_do_cong_thuc but NOT applied; the s0_over rule still applies."""
    sd = sl.sample_size(_q({('ADRC', 'ba_mu'): 0}), 1000, 60.0, False, tool_n=500)
    assert sd['chay'] is True and sd['N_da_dung'] == 500 and sd['ly_do'] == []
    assert any('q_min' in r for r in sd['ly_do_cong_thuc'])      # formula reasons recorded
    sd = sl.sample_size(_q({}), 1000, 28800.0 + 1.0, False, tool_n=500)
    assert sd['chay'] is True and sd['N_da_dung'] == 500         # N = 0 does not apply
    assert any('< 3000' in r for r in sd['ly_do_cong_thuc'])
    sd = sl.sample_size(_q({}), 1000, 60.0, True, tool_n=500)
    assert sd['chay'] is False and sd['N_da_dung'] == 0          # s0_over still applies


# ── fake workers for run_step ───────────────────────────────────────────────────────

def _fake_gate(e_mid=0.05, e_rest=None, bad=None, second_e=None, second_err=False, tv_nan=None):
    """A deterministic safe_run_gate: a feasible 'ok' row (every scored field below its bound)
    with e_perp_max = e_mid at the middle mu (e_rest elsewhere), or a 'run_error' row when
    bad(g, v, mu) holds, or an 'ok' row with tv_duty NaN when tv_nan(g, v, mu) holds (the
    evaluator turns it into a run_error, S04 tv_duty_run_error). gate.second flags the
    determinism rerun (item 12 tests)."""
    def gate(fw, path, ctrl, g, v, mu):
        if gate.second and second_err:
            return dict(status='run_error', completed='', error='fake rerun run_error'), None
        e = e_mid if mu == MU_MID else (e_rest if e_rest is not None else e_mid)
        if gate.second and second_e is not None:
            e = second_e
        if bad is not None and bad(g, v, mu):
            return dict(status='run_error', completed='', error='fake run_error'), None
        tv = float('nan') if tv_nan is not None and tv_nan(g, v, mu) else 0.4
        kep = dict(kep_frac_vx=0.0, kep_frac_vy=0.0, kep_frac_wz=0.0, kep_note='') \
            if ctrl == 'PID' else dict(kep_frac_vx='', kep_frac_vy='', kep_frac_wz='', kep_note='')
        return dict(status='ok', completed='true', e_perp_max=e, tv_duty=tv, sat_frac=0.0,
                    t_end=1.0, e_theta_lane_max_deg=0.5, e_theta_cmd_max_deg=0.5, ds_absmax=1.0,
                    e_trk_max=0.05, e_bao_max=0.02, **kep), {'t': np.arange(3.0)}  # C3-48
    gate.second = False
    return gate


def _fake_new_process(dll, runs, work, timeout=None):     # C3-40: run_in_new_process takes timeout
    os.makedirs(work, exist_ok=True)
    return [dict(row=dict(status='ok', completed='true', e_perp_max=0.05, tv_duty=0.4,
                          sat_frac=0.0, t_end=1.0, e_theta_lane_max_deg=0.5,
                          e_theta_cmd_max_deg=0.5, ds_absmax=1.0, e_trk_max=0.05, e_bao_max=0.02),
                 digest=s3.log_digest({'t': np.arange(3.0)})) for _ in runs]


def _patch(monkeypatch, gate, check=None):
    """run_jobs in-process over sl.do_job (the same scoring path); check() runs per job batch."""
    monkeypatch.setattr(s4, 'safe_run_gate', gate)
    monkeypatch.setattr(sl, '_firmware', lambda dll: object())
    monkeypatch.setattr(s3, 'run_in_new_process', _fake_new_process)
    monkeypatch.setattr(sl, '_clock', itertools.count().__next__)

    def jobs(jobs_, work, n, lp):
        if check is not None:
            check(jobs_)
        if jobs_ and jobs_[0]['name'].startswith('kiem_tat_dinh'):
            gate.second = True
        return [sl.do_job(j) for j in jobs_]
    monkeypatch.setattr(sl, 'run_jobs', jobs)


def _screen(n0, n, us):
    """us: {set: u value or array}; the four sets run_step reads."""
    def arr(x, m):
        return np.full((m, 7), x) if isinstance(x, float) else np.asarray(x, dtype=float)
    return dict(S0=arr(us.get('S0', 0.5), n0), S1=arr(us['S1'], n), S2=arr(us['S2'], n),
                S2b=arr(us['S2b'], n))


def _run(tmp_path, monkeypatch, screen, gate, speeds=(0.5,), check=None, tool='default'):
    _patch(monkeypatch, gate, check)
    W = _W()
    if tool == 'default':
        tool = dict(n0=len(screen['S0']), n=len(screen['S1']))
    return sl.run_step(str(tmp_path), 'fake.dll', s3.CONTROLLERS, list(speeds), MU, W, _quet(W),
                       screen, 2, tool_check=tool), W


def _ket_qua(tmp_path):
    with open(tmp_path / 'sangloc' / 'ket_qua.json', encoding='utf-8') as f:
        return json.load(f)


def _read_csv(path):
    import csv
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def test_run_step_nhan(tmp_path, monkeypatch, capsys):
    """items 4-12: every set feasible and inside -> both modes accepted at check 1, the domain is
    B2 = B1 (B_j widened one node each side of the single node all gains sit on), nothing is
    printed and every file lies under sealed/."""
    res, W = _run(tmp_path, monkeypatch, _screen(40, 100, dict(S1=0.5, S2=0.5, S2b=0.5)),
                  _fake_gate())
    assert res['nhanh'] == {'PID': 'nhan', 'ADRC': 'nhan'} and res['ly_do'] == {'PID': [], 'ADRC': []}
    assert res['N_da_dung'] == 100
    sl_dir = tmp_path / 'sangloc'
    for n in ('points_S1.csv', 'evals_S1.csv', 'score_S1.csv', 'khoa_S1.json', 'points_S2.csv',
              'evals_S2.csv', 'score_S2.csv', 'kiem_S2.json', 'points_S2b.csv', 'evals_S2b.csv',
              'score_S2b.csv', 'kiem_S2b.json', 'determinism_sangloc.csv', 'ket_qua.json'):
        assert (sl_dir / n).is_file(), n
    for n in ('points_S0.csv', 'evals_S0.csv', 'score_S0.csv', 'so_dem.json'):
        assert (tmp_path / 'sangloc_thu' / n).is_file(), n
    doc = _ket_qua(tmp_path)
    assert doc['N_da_dung'] == 100 and doc['m_min'] == 86
    for ctrl in s3.CONTROLLERS:
        c = doc[ctrl]
        assert c['nhanh'] == 'nhan' and c['nguon_mien'] == 'sang_loc' and c['ly_do'] == []
        for f_ in s3.FACTORS[ctrl]:
            ns = W[ctrl][f_]
            x = ns[0] * (ns[-1] / ns[0]) ** 0.5                # the gain every row reaches
            lo = max(n for n in ns if n <= x)
            hi = min(n for n in ns if n >= x)
            if lo == hi:                                       # on a node: widen one each side
                i = ns.index(lo)
                lo = ns[i - 1] if i > 0 else lo
                hi = ns[i + 1] if i + 1 < len(ns) else hi
            assert c['mien'][f_] == [float(lo), float(hi)]
            assert c['hop']['B1'][f_] == c['mien'][f_] == c['hop']['B2'][f_]
        for m in sl.MODES:
            assert c['che_do'][m]['kiem_duoc_nhan_o_lan'] == 1
            assert c['che_do'][m]['can'] == pytest.approx(1 - 0.0125 ** (1 / 100))
            assert c['che_do'][m]['tap']['S1']['n_thuoc_C'] == 100
            assert c['che_do'][m]['tap']['S2']['m'] == 100 and c['che_do'][m]['tap']['S2']['k'] == 0
        assert c['n_ngoai_bang'] == {'S1': 0, 'S2': 0, 'S2b': 0}
        assert c['n_cap_trung_float32'] == 300 * 299 // 2   # every row has the same seven f32
        assert c['ti_le_kha_thi_trong_mien']['giua']['ti_le'] == 1.0
    khoa = json.load(open(sl_dir / 'khoa_S1.json', encoding='utf-8'))
    assert khoa['N'] == 100 and set(khoa['sha256']) == {'points_S1.csv', 'evals_S1.csv', 'score_S1.csv'}
    kiem = json.load(open(sl_dir / 'kiem_S2.json', encoding='utf-8'))
    l1 = kiem['PID']['lan_1']['giua']
    assert l1['m'] == 100 and l1['k'] == 0 and l1['duoc_nhan'] is True
    assert l1['can'] == pytest.approx(1 - 0.0125 ** (1 / 100))
    det = _read_csv(sl_dir / 'determinism_sangloc.csv')
    assert {r['row'] for r in det} == {'1', '2', '3', '4', '5'}
    assert all(r['log_arrays_equal'] == 'true' and r['fields_equal'] == 'true' for r in det)
    pha = {p['pha']: p for p in res['pha']}
    assert list(pha) == ['S0', 'S1', 'S2', 'S2b', 'kiem_tat_dinh_sang_loc']
    assert pha['S1']['n_luot_thuc'] == 600 and pha['S1']['n_luot_thuc_theo_controller'] == \
        {'PID': 300, 'ADRC': 300}                                   # 3 runs of every set
    out, err = capsys.readouterr()
    assert out == '' and err == ''                                   # the step is silent


def test_run_step_widening(tmp_path, monkeypatch):
    """item 8: a mode not accepted at check 1 (S2 has C_j sets outside B1) widens B2 over those
    sets; check 2 on S2b accepts."""
    u_s2 = np.full((100, 7), 0.5)
    u_s2[-10:] = 0.9                                                # 10 C_j sets outside B1
    u_s2b = np.full((100, 7), 0.5)
    u_s2b[::2] = 0.9                                                # inside B2 afterwards
    res, W = _run(tmp_path, monkeypatch, _screen(40, 100, dict(S1=0.5, S2=u_s2, S2b=u_s2b)),
                  _fake_gate())
    doc = _ket_qua(tmp_path)
    assert res['nhanh']['PID'] == 'nhan'
    c = doc['PID']
    for m in sl.MODES:
        assert c['che_do'][m]['kiem_duoc_nhan_o_lan'] == 2
    kiem = json.load(open(tmp_path / 'sangloc' / 'kiem_S2.json', encoding='utf-8'))
    assert kiem['PID']['lan_1']['giua']['duoc_nhan'] is False and kiem['PID']['lan_1']['giua']['k'] == 10
    for f_ in s3.FACTORS['PID']:
        ns = W['PID'][f_]
        x = ns[0] * (ns[-1] / ns[0]) ** 0.5                            # the u = 0.5 gains
        lo, hi = max(n for n in ns if n <= x), min(n for n in ns if n >= x)
        if lo == hi:
            i = ns.index(lo)
            lo, hi = (ns[i - 1] if i > 0 else lo), (ns[i + 1] if i + 1 < len(ns) else hi)
        assert c['hop']['B1'][f_] == [float(lo), float(hi)]
        assert c['hop']['B2'][f_] == [float(lo), float(ns[-1])]        # widened to the last node
        assert c['mien'][f_] == c['hop']['B2'][f_]
    kiem_b = json.load(open(tmp_path / 'sangloc' / 'kiem_S2b.json', encoding='utf-8'))
    assert kiem_b['PID']['lan_2']['giua']['vai_tro'] == 'quyet_dinh'
    assert kiem_b['PID']['lan_2']['giua']['duoc_nhan'] is True


def test_run_step_du_phong_kiem(tmp_path, monkeypatch):
    """item 8/9: boxes exist but a mode is accepted by neither check (m < m_min) -> du_phong_kiem
    and the fallback domain."""
    res, W = _run(tmp_path, monkeypatch, _screen(40, 50, dict(S1=0.5, S2=0.5, S2b=0.5)),
                  _fake_gate())
    doc = _ket_qua(tmp_path)
    for ctrl in s3.CONTROLLERS:
        c = doc[ctrl]
        assert c['nhanh'] == 'du_phong_kiem' and c['nguon_mien'] == 'du_phong' and c['mien'] is None
        assert all(r.startswith('du_phong_kiem') for r in c['ly_do'])
        for m in sl.MODES:
            assert c['che_do'][m]['kiem_duoc_nhan_o_lan'] is None
    kiem = json.load(open(tmp_path / 'sangloc' / 'kiem_S2.json', encoding='utf-8'))
    assert kiem['PID']['lan_1']['giua'] == dict(m=50, k=0, duoc_nhan=False, can=1 - 0.0125 ** (1 / 50))
    assert res['mien'] == {'PID': None, 'ADRC': None} and res['N_da_dung'] == 50


def test_run_step_du_phong_khong_nguong(tmp_path, monkeypatch):
    """items 6, 9: ba_mu without a threshold (S1 had no feasible set in it) -> no box, the
    fallback domain, but S2 and S2b still run and are scored in full."""
    res, W = _run(tmp_path, monkeypatch, _screen(40, 100, dict(S1=0.5, S2=0.5, S2b=0.5)),
                  _fake_gate(e_mid=0.05, e_rest=0.5))              # infeasible at low/high mu
    doc = _ket_qua(tmp_path)
    for ctrl in s3.CONTROLLERS:
        c = doc[ctrl]
        assert c['nhanh'] == 'du_phong_khong_nguong' and c['mien'] is None
        assert c['nguong']['ba_mu'] is None and c['nguong']['giua'] is not None
        assert all(c['hop'][b] is None for b in ('B_giua', 'B_ba_mu', 'B1', 'B2'))
        assert all(r.startswith('du_phong_khong_nguong') for r in c['ly_do'])
        assert c['che_do']['giua']['tap']['S1']['n_thuoc_C'] == 100
        assert c['che_do']['ba_mu']['tap']['S1']['n_thuoc_C'] is None
        assert c['che_do']['ba_mu']['tap']['S2']['m'] is None \
            and c['che_do']['ba_mu']['tap']['S2']['k'] is None
    ev = _read_csv(tmp_path / 'sangloc' / 'evals_S2.csv')           # S2 ran in full (3 rows each)
    assert len(ev) == 600
    sc = _read_csv(tmp_path / 'sangloc' / 'score_S2.csv')
    assert all(r['kha_thi'] == 'false' for r in sc if r['che_do'] == 'ba_mu')


def test_run_step_du_phong_loi_chay(tmp_path, monkeypatch):
    """items 5, 9: a set over 10 % unclassifiable in a mode -> the fallback domain, even when the
    checks pass (the boxes and both checks are still computed and reported)."""
    u_s2 = np.full((100, 7), 0.5)
    u_s2[-11:] = 0.9                                                # 11 run_error rows of 100
    res, W = _run(tmp_path, monkeypatch, _screen(40, 100, dict(S1=0.5, S2=u_s2, S2b=0.5)),
                  _fake_gate(bad=lambda g, v, mu: g['pos_kp'] > 50.0))
    doc = _ket_qua(tmp_path)
    for ctrl in s3.CONTROLLERS:
        c = doc[ctrl]
        assert c['nhanh'] == 'du_phong_loi_chay' and c['mien'] is None
        assert any(r.startswith('du_phong_loi_chay') and 'S2' in r for r in c['ly_do'])
        assert c['che_do']['giua']['kiem_duoc_nhan_o_lan'] == 1      # checks still ran
        assert c['che_do']['giua']['tap']['S2']['n_phan_loai_duoc'] == 89


def test_run_step_reasons_and_priority(tmp_path, monkeypatch):
    """item 9: several reasons -> ly_do lists all of them (each starting with its branch name)
    and nhanh follows BRANCH_PRIORITY."""
    u_s1 = np.full((100, 7), 0.5)
    u_s1[:15] = 0.9                                                 # 15 run_error rows of S1
    res, W = _run(tmp_path, monkeypatch, _screen(40, 100, dict(S1=u_s1, S2=0.5, S2b=0.5)),
                  _fake_gate(e_mid=0.05, e_rest=0.5, bad=lambda g, v, mu: g['pos_kp'] > 50.0))
    c = _ket_qua(tmp_path)['ADRC']
    kinds = {r.split(':')[0] for r in c['ly_do']}
    assert kinds == {'du_phong_khong_nguong', 'du_phong_loi_chay'}
    assert c['nhanh'] == 'du_phong_khong_nguong'                     # priority over loi_chay
    assert res['nhanh']['ADRC'] == 'du_phong_khong_nguong'


def test_run_step_khong_chay_sang_loc(tmp_path, monkeypatch):
    """item 10: no feasible S0 set -> q_min = 0 -> N q_min < m_min -> no screening: ket_qua.json
    only (no file of S1, S2, S2b), N_da_dung 0, every controller khong_chay_sang_loc."""
    screen = dict(S0=np.full((1000, 7), 0.5), S1=np.full((4, 7), 0.5), S2=np.full((4, 7), 0.5),
                  S2b=np.full((4, 7), 0.5))
    res, W = _run(tmp_path, monkeypatch, screen, _fake_gate(e_mid=0.5), tool=None)  # all infeasible
    assert res['N_da_dung'] == 0 and res['pha'][0]['pha'] == 'S0'
    assert res['nhanh'] == {'PID': 'khong_chay_sang_loc', 'ADRC': 'khong_chay_sang_loc'}
    doc = _ket_qua(tmp_path)
    for ctrl in s3.CONTROLLERS:
        c = doc[ctrl]
        assert c['nhanh'] == 'khong_chay_sang_loc' and c['mien'] is None and c['ly_do']
    assert not (tmp_path / 'sangloc' / 'points_S1.csv').exists()
    assert not (tmp_path / 'sangloc' / 'evals_S2.csv').exists()
    assert not (tmp_path / 'sangloc' / 'score_S2b.csv').exists()
    sd = json.load(open(tmp_path / 'sangloc_thu' / 'so_dem.json', encoding='utf-8'))
    assert sd['chay'] is False and sd['q_min'] == 0.0 and sd['N_da_dung'] == 0


def test_run_step_ngoai_bang_rows(tmp_path, monkeypatch):
    """items 4, 5, 12: ngoai_bang rows are never run (no eval rows), stay in the points and score
    files, are counted per set in ket_qua.json, and are left out of the phase run counts and the
    determinism rows."""
    u_s1 = np.full((100, 7), 0.5)
    u_s1[90:] = 0.99                                              # ADRC rows 91-100: triangle
    res, W = _run(tmp_path, monkeypatch, _screen(40, 100, dict(S1=u_s1, S2=0.5, S2b=0.5)),
                  _fake_gate())
    doc = _ket_qua(tmp_path)
    assert doc['ADRC']['n_ngoai_bang'] == {'S1': 10, 'S2': 0, 'S2b': 0}
    assert doc['ADRC']['che_do']['giua']['tap']['S1']['n_phan_loai_duoc'] == 100
    points = _read_csv(tmp_path / 'sangloc' / 'points_S1.csv')
    nb = [r for r in points if r['ngoai_bang'] == 'true']
    assert len(nb) == 10 and all(r['controller'] == 'ADRC' and int(r['row']) >= 91 for r in nb)
    evals = _read_csv(tmp_path / 'sangloc' / 'evals_S1.csv')
    assert not [r for r in evals if r['controller'] == 'ADRC' and int(r['row']) >= 91]
    scores = _read_csv(tmp_path / 'sangloc' / 'score_S1.csv')
    s_nb = [r for r in scores if r['controller'] == 'ADRC' and int(r['row']) == 91]
    assert all(r['ngoai_bang'] == 'true' and r['status'] == 'ngoai_bang'
               and r['phan_loai_duoc'] == 'true' and r['kha_thi'] == 'false' for r in s_nb)
    pha = {p['pha']: p for p in res['pha']}
    assert pha['S1']['n_luot_thuc'] == 300 + 90 * 3                  # no runs for ngoai_bang
    det = _read_csv(tmp_path / 'sangloc' / 'determinism_sangloc.csv')
    assert {r['row'] for r in det} == {'1', '2', '3', '4', '5'}      # first five runnable rows


def test_run_step_lock_order(tmp_path, monkeypatch):
    """section 7 order: khoa_S1.json and the three S1 files exist before the first S2 job;
    kiem_S2.json exists before the first S2b job."""
    seen = []

    def check(jobs):
        sl_dir = tmp_path / 'sangloc'
        names = [j['name'] for j in jobs]
        if any(n.startswith('S2_') for n in names) and 'S2' not in seen:
            seen.append('S2')
            assert (sl_dir / 'khoa_S1.json').is_file()
            for n in ('points_S1.csv', 'evals_S1.csv', 'score_S1.csv'):
                assert (sl_dir / n).is_file(), n
        if any(n.startswith('S2b_') for n in names) and 'S2b' not in seen:
            seen.append('S2b')
            assert (sl_dir / 'kiem_S2.json').is_file()

    _run(tmp_path, monkeypatch, _screen(40, 100, dict(S1=0.5, S2=0.5, S2b=0.5)),
         _fake_gate(), check=check)
    assert seen == ['S2', 'S2b']


def test_run_step_two_gate_speeds(tmp_path, monkeypatch):
    """item 5: g = 2 speeds -> 2 runs at the middle mu and 4 more when the set is feasible."""
    res, W = _run(tmp_path, monkeypatch, _screen(40, 50, dict(S1=0.5, S2=0.5, S2b=0.5)),
                  _fake_gate(), speeds=(0.5, 0.8))
    ev = _read_csv(tmp_path / 'sangloc' / 'evals_S1.csv')
    assert len(ev) == 2 * 50 * 6                                    # 2 controllers, 6 runs a set
    sc = _read_csv(tmp_path / 'sangloc' / 'score_S1.csv')
    assert all(r['n_runs'] == '2' for r in sc if r['che_do'] == 'giua')
    assert all(r['n_runs'] == '6' for r in sc if r['che_do'] == 'ba_mu')


def test_run_step_determinism_difference(tmp_path, monkeypatch):
    """item 12: a rerun giving a different value (or a run_error) stops the step before
    ket_qua.json; determinism_sangloc.csv is still written."""
    with pytest.raises(s3.S03Stop, match='determinism'):
        _run(tmp_path, monkeypatch, _screen(40, 100, dict(S1=0.5, S2=0.5, S2b=0.5)),
             _fake_gate(second_e=0.07))
    assert (tmp_path / 'sangloc' / 'determinism_sangloc.csv').is_file()
    assert not (tmp_path / 'sangloc' / 'ket_qua.json').exists()
    det = _read_csv(tmp_path / 'sangloc' / 'determinism_sangloc.csv')
    assert any(r['fields_equal'] == 'false' for r in det)


def test_run_step_determinism_run_error(tmp_path, monkeypatch):
    """item 12: a run_error on the rerun counts as a difference."""
    with pytest.raises(s3.S03Stop, match='determinism'):
        _run(tmp_path, monkeypatch, _screen(40, 100, dict(S1=0.5, S2=0.5, S2b=0.5)),
             _fake_gate(second_err=True))
    assert not (tmp_path / 'sangloc' / 'ket_qua.json').exists()


def test_run_step_determinism_rows_skip(tmp_path, monkeypatch):
    """item 12: the determinism rows are the first five S1 rows that are not ngoai_bang and have
    no run_error run: ADRC rows 1-3 (ngoai_bang) and row 4 (run_error) are skipped -> 5..9; PID
    rows 1-3 run fine, row 4 is a run_error -> 1, 2, 3, 5, 6."""
    u_s1 = np.full((100, 7), 0.5)
    u_s1[0:3] = 0.99                          # ADRC ngoai_bang; PID pos_kp ~142: runs fine
    u_s1[3] = 0.9                             # pos_kp ~91: run_error for both controllers
    res, W = _run(tmp_path, monkeypatch, _screen(40, 100, dict(S1=u_s1, S2=0.5, S2b=0.5)),
                  _fake_gate(bad=lambda g, v, mu: 60.0 < g['pos_kp'] < 100.0))
    doc = _ket_qua(tmp_path)
    assert doc['ADRC']['kiem_tat_dinh']['dong'] == [5, 6, 7, 8, 9]
    assert doc['PID']['kiem_tat_dinh']['dong'] == [1, 2, 3, 5, 6]
    assert doc['ADRC']['kiem_tat_dinh']['ket_qua'] == 'dat'
    det = _read_csv(tmp_path / 'sangloc' / 'determinism_sangloc.csv')
    assert {r['row'] for r in det if r['controller'] == 'ADRC'} == {'5', '6', '7', '8', '9'}
    assert {r['row'] for r in det if r['controller'] == 'PID'} == {'1', '2', '3', '5', '6'}


def test_run_step_tv_duty_real_path(tmp_path, monkeypatch):
    """item 5: an 'ok' run with tv_duty NaN becomes a run_error through s4.tv_duty_run_error at
    the run level (attempt 1 'run_error' with the tv_duty reason, the diagnosis rerun exists as
    attempt 2), the set is unclassifiable in both modes and counted in the 10 % numerator."""
    u_s1 = np.full((100, 7), 0.5)
    u_s1[:5] = 0.9                            # pos_kp ~91: the tv_nan rows
    res, W = _run(tmp_path, monkeypatch, _screen(40, 100, dict(S1=u_s1, S2=0.5, S2b=0.5)),
                  _fake_gate(tv_nan=lambda g, v, mu: 60.0 < g['pos_kp'] < 100.0))
    evals = _read_csv(tmp_path / 'sangloc' / 'evals_S1.csv')
    err = [r for r in evals
           if r['controller'] == 'PID' and r['row'] == '1' and r['attempt'] == '1']
    assert len(err) == 1 and err[0]['status'] == 'run_error' and 'tv_duty' in err[0]['error']
    assert any(r['controller'] == 'PID' and r['row'] == '1' and r['attempt'] == '2'
               for r in evals)
    scores = _read_csv(tmp_path / 'sangloc' / 'score_S1.csv')
    for r in scores:
        if r['controller'] == 'PID' and r['row'] == '1':
            assert r['phan_loai_duoc'] == 'false'
    doc = _ket_qua(tmp_path)
    for m in sl.MODES:
        t = doc['PID']['che_do'][m]['tap']['S1']
        assert t['n_bo_gain'] - t['n_phan_loai_duoc'] == 5   # in the 10 % numerator
    pha = {p['pha']: p for p in res['pha']}
    assert pha['S1']['n_chay_lai_chan_doan'] > 0


def test_run_step_error_limit_boundary(tmp_path, monkeypatch):
    """items 5, 10: exactly 10 % unclassifiable is not over the limit -> the step stays accepted
    (one more, tested in test_run_step_du_phong_loi_chay, is du_phong_loi_chay)."""
    u_s2 = np.full((100, 7), 0.5)
    u_s2[-10:] = 0.9                          # exactly 10 run_error rows of 100
    res, W = _run(tmp_path, monkeypatch, _screen(40, 100, dict(S1=0.5, S2=u_s2, S2b=0.5)),
                  _fake_gate(bad=lambda g, v, mu: g['pos_kp'] > 50.0))
    doc = _ket_qua(tmp_path)
    for ctrl in s3.CONTROLLERS:
        c = doc[ctrl]
        assert c['nhanh'] == 'nhan'
        assert not any(r.startswith('du_phong_loi_chay') for r in c['ly_do'])
        t = c['che_do']['giua']['tap']['S2']
        assert t['n_bo_gain'] - t['n_phan_loai_duoc'] == 10  # exactly 10 %: still accepted


def test_run_step_s0_ngoai_bang_q(tmp_path, monkeypatch):
    """items 5, 10: ngoai_bang S0 rows keep the q denominator mau = n0 and are counted in
    n_ngoai_bang of so_dem.json."""
    u_s0 = np.full((40, 7), 0.5)
    u_s0[:5] = 0.99                           # ADRC rows 1-5: ngoai_bang, never run
    res, W = _run(tmp_path, monkeypatch,
                  dict(S0=u_s0, S1=np.full((100, 7), 0.5), S2=np.full((100, 7), 0.5),
                       S2b=np.full((100, 7), 0.5)),
                  _fake_gate(), tool=dict(n0=40, n=100))
    sd = json.load(open(tmp_path / 'sangloc_thu' / 'so_dem.json', encoding='utf-8'))
    assert sd['dem']['ADRC']['n_ngoai_bang'] == 5 and sd['dem']['PID']['n_ngoai_bang'] == 0
    assert sd['q']['ADRC giua']['mau'] == 40 and sd['q']['ADRC giua']['tu'] == 35
    assert sd['nhanh_dieu_10'] == 'chay'
    ev = _read_csv(tmp_path / 'sangloc_thu' / 'evals_S0.csv')
    assert not [r for r in ev if r['controller'] == 'ADRC' and int(r['row']) <= 5]


def test_run_step_input_validation(tmp_path, monkeypatch):
    """items 3, 5: two or unsorted mu levels, n0 over the S0 rows, N over the set rows -> S03Stop
    before any run."""
    screen = _screen(40, 100, dict(S1=0.5, S2=0.5, S2b=0.5))
    W, quet = _W(), _quet(_W())
    for mus in ([0.4, 0.5], [0.473, 0.413, 0.436]):
        with pytest.raises(s3.S03Stop):
            sl.run_step(str(tmp_path), 'd.dll', s3.CONTROLLERS, [0.5], mus, W, quet, screen, 2)
    with pytest.raises(s3.S03Stop):
        sl.run_step(str(tmp_path), 'd.dll', s3.CONTROLLERS, [0.5], MU, W, quet, screen, 2,
                    tool_check=dict(n0=41, n=100))
    with pytest.raises(s3.S03Stop):
        sl.run_step(str(tmp_path), 'd.dll', [], [0.5], MU, W, quet, screen, 2)
    with pytest.raises(s3.S03Stop):          # N over the S1 rows, found after S0 ran
        _run(tmp_path, monkeypatch, screen, _fake_gate(), tool=dict(n0=40, n=101))


def test_skip_step(tmp_path):
    """skip_step writes only sealed/sangloc/ket_qua.json: every controller khong_chay_sang_loc,
    trang_thai 'hoan_tat', the one reason, N null, N_da_dung 0; same dict shape as run_step."""
    res = sl.skip_step(str(tmp_path), s3.CONTROLLERS, 'khong_chay_sang_loc: mot muc mu')
    assert res['mien'] == {'PID': None, 'ADRC': None}
    assert res['nhanh'] == {'PID': 'khong_chay_sang_loc', 'ADRC': 'khong_chay_sang_loc'}
    assert res['N_da_dung'] == 0 and res['pha'] == []
    doc = _ket_qua(tmp_path)
    assert doc['N'] is None and doc['N_da_dung'] == 0
    for ctrl in s3.CONTROLLERS:
        assert doc[ctrl]['nhanh'] == 'khong_chay_sang_loc' and doc[ctrl]['mien'] is None
        assert doc[ctrl]['trang_thai'] == 'hoan_tat'
        assert doc[ctrl]['ly_do'] == ['khong_chay_sang_loc: mot muc mu']
    files = [str(p.relative_to(tmp_path)) for p in tmp_path.rglob('*') if p.is_file()]
    assert files == [os.path.join('sangloc', 'ket_qua.json')]


# ── the sealed levels file (section 7) ─────────────────────────────────────────────

def _grid_and_W(ctrl):
    """Real sweep grids and W nodes of the fake defaults."""
    defaults = FAKE_DEFAULTS[ctrl]
    hi = {'PID': dict(kp_xy=50.0, ki_xy=200.0, kp_wz=5.0, ki_wz=100.0, pos_kp=50.0, pos_ki=100.0,
                      k_psi=10.0),
          'ADRC': dict(omega_c_xy=150.0, omega_o_xy=48.0, omega_c_wz=150.0, omega_o_wz=48.0,
                       pos_kp=50.0, pos_ki=100.0, k_psi=10.0)}[ctrl]
    gi, W = {}, {}
    for n in s3.FACTORS[ctrl]:
        _, gi[(ctrl, n)] = s3.sweep_grid_cut(ctrl, n, defaults[n], 0.5, hi[n])
        W[n] = s3.w_nodes(ctrl, n, defaults[n], 0.5, hi[n])
    return gi, W, defaults


def test_levels_rows_labels_and_scale():
    """section 7: L_nguon 'biên W' at the first node, 'sàng lọc' inside; H_nguon 'trần bảng chặn'
    over 'biên W' when H = b_n = T, 'trần ω_o' for omega_o at 48, 'biên W' else, 'sàng lọc'
    below the last node; log scale when H/L >= 4 else lin."""
    gi, W, defaults = _grid_and_W('PID')
    mien = {n: (W[n][0], W[n][-1]) for n in W}                    # the whole W box
    rows = {r['factor']: r for r in sl.levels_rows('PID', mien, {'PID': W}, gi, defaults, [], 'nhan', [],
                                                  [0.5])}
    r = rows['kp_xy']                                             # b_n = T = 41 after the cut
    assert r['L_nguon'] == 'biên W' and r['H_nguon'] == 'trần bảng chặn' and r['scale'] == 'log'
    r = rows['pos_ki']                                            # b_n < T at the last node
    assert r['H_nguon'] == 'biên W' and r['nguon_mien'] == 'sang_loc' and r['ly_do'] == ''
    mien2 = {n: (W[n][1], W[n][-2]) for n in W}                   # strictly inside W
    rows2 = {r['factor']: r for r in sl.levels_rows('PID', mien2, {'PID': W}, gi, defaults, [], 'nhan', [],
                                                   [0.5])}
    assert rows2['kp_xy']['L_nguon'] == 'sàng lọc' and rows2['kp_xy']['H_nguon'] == 'sàng lọc'
    mien3 = dict(mien, kp_xy=(30.0, 41.0))                        # H/L < 4 -> lin
    rows3 = {r['factor']: r for r in sl.levels_rows('PID', mien3, {'PID': W}, gi, defaults, [], 'nhan', [],
                                                   [0.5])}
    assert rows3['kp_xy']['scale'] == 'lin'
    gi_a, W_a, defaults_a = _grid_and_W('ADRC')
    mien_a = {n: (W_a[n][0], W_a[n][-1]) for n in W_a}
    rows_a = {r['factor']: r for r in sl.levels_rows('ADRC', mien_a, {'ADRC': W_a}, gi_a, defaults_a, [],
                                                    'nhan', [], [0.5])}
    assert rows_a['omega_o_xy']['H_nguon'] == 'trần ω_o'          # H = 48, T = 100: no bang cut


def test_levels_rows_fallback_and_readback(tmp_path):
    """section 7, item 9: fallback copies the public levels.csv rows (L = 0 kept) with
    nguon_mien 'du_phong' and the reasons; read_levels gives the floats back bit-exact."""
    gi, W, defaults = _grid_and_W('PID')
    pub = [dict(controller='PID', factor='kp_xy', L=0.0, H=50.0, centre=25.0, scale='lin',
                default=defaults['kp_xy'], rule_note='public', a=0.15, b=50.0, b_n=41.0,
                L_nguon='biên miền', H_nguon='trần bảng chặn')]
    rows = sl.levels_rows('PID', None, {'PID': W}, gi, defaults, pub, 'du_phong_kiem',
                          ['du_phong_kiem: che do giua ...'], [0.5])
    assert len(rows) == 1 and rows[0]['L'] == 0.0 and rows[0]['nguon_mien'] == 'du_phong'
    assert rows[0]['ly_do'] == 'du_phong_kiem: che do giua ...'
    mien = {n: (W[n][1], W[n][-1]) for n in W}
    all_rows = sl.levels_rows('PID', mien, {'PID': W}, gi, defaults, [], 'nhan', [], [0.5])
    p = str(tmp_path / 'levels_sangloc.csv')
    s3.write_csv(p, all_rows, sl.LEVEL_SL_COLS)
    back = sl.read_levels(p, 'PID')
    for n in s3.FACTORS['PID']:
        L, H, sc = back[n]
        assert _bits(L) == _bits(mien[n][0]) and _bits(H) == _bits(mien[n][1])
        assert sc == s3.scale_of(*mien[n])[0]


# ── the worker processes (item 5) ───────────────────────────────────────────────────

def test_run_jobs_worker_failure(tmp_path):
    """A child that exits non-zero (here: the DLL path does not exist) makes run_jobs a S03Stop;
    the child log is kept at log_path and no job file is left."""
    work = str(tmp_path / '_work')
    log_path = str(tmp_path / 'tien_trinh.log')
    g = dict(FAKE_DEFAULTS['PID'])
    job = dict(name='badjob', dll=str(tmp_path / 'no_such.dll'), controller='PID', speeds=[0.5],
               mu_levels=MU, rows=[(1, g)], work=os.path.join(work, 'badjob'))
    with pytest.raises(s3.S03Stop, match='worker'):
        sl.run_jobs([job], work, 2, log_path)
    text = open(log_path, encoding='utf-8', errors='replace').read()
    assert 'badjob' in text                                       # headed by the job name
    assert not os.path.exists(work) or not os.listdir(work)       # no job file is left


def test_run_jobs_three_workers_failure(tmp_path):
    """item 5: one failing child of three running at once ends run_jobs in a S03Stop (not a
    KeyError of the scheduler loop); the log stays at log_path, no job file is left."""
    work = str(tmp_path / '_work')
    log_path = str(tmp_path / 'tien_trinh.log')
    g = dict(FAKE_DEFAULTS['PID'])
    jobs = [dict(name=f'bad{i}', dll=str(tmp_path / 'no_such.dll'), controller='PID',
                 speeds=[0.5], mu_levels=MU, rows=[(1, g)],
                 work=os.path.join(work, f'bad{i}')) for i in range(3)]
    with pytest.raises(s3.S03Stop, match='worker'):
        sl.run_jobs(jobs, work, 3, log_path)
    assert not os.path.exists(work) or not os.listdir(work)
    assert 'bad' in open(log_path, encoding='utf-8', errors='replace').read()


@needs_dll
def test_run_set_real_runs_deterministic(tmp_path):
    """item 5 + the item-12 property: two PID gain sets (one the firmware defaults) at one gate
    speed and the three mu levels, twice in new processes: same log SHA-256 and the same fields,
    every eval row status 'ok' with the columns of EVAL_SL_COLS."""
    fw = fwsim.FirmwareSim(_DLL)
    g0 = s3.defaults(fw, 'PID')
    g1 = dict(g0, kp_xy=g0['kp_xy'] * 1.5)
    rows = dict(PID=[dict(row=1, u=[0.0] * 7, g=g0, flag='ok'),
                     dict(row=2, u=[0.0] * 7, g=g1, flag='ok')])
    res1, _ = sl.run_set('T', rows, _DLL, [0.8], MU, str(tmp_path / 'w1'), 2,
                         str(tmp_path / 'tien_trinh_1.log'))
    res2, _ = sl.run_set('T', rows, _DLL, [0.8], MU, str(tmp_path / 'w2'), 2,
                         str(tmp_path / 'tien_trinh_2.log'))
    want = s4.EVAL_COLS[s4.EVAL_COLS.index('status'):s4.EVAL_COLS.index('e_bao_max') + 1] + \
        [k for k in s4.EVAL_COLS if k.startswith('pen_')]
    for rn in (1, 2):
        a, b = res1['PID'][rn], res2['PID'][rn]
        assert a and len(a) == len(b)
        for e in a:
            assert e['status'] == 'ok'
            assert isinstance(e['log_sha256'], str) and len(e['log_sha256']) == 64
            assert all(k in e for k in want)
        for e1, e2 in zip(a, b):
            assert e1['log_sha256'] == e2['log_sha256']
            assert all(s3._bits(e1.get(k)) == s3._bits(e2.get(k)) for k in sl.EVAL_SL_COLS)


# ── C3-36: the numbers file is checked against its generation source ─────────────

def test_c3_36_verify_screen_source_unit(monkeypatch):
    """C3-36: generate_sets / canonical_text / verify_screen_source — the generated sets pass
    (bit for bit and as text); one ulp moved, a sign-of-zero, a missing set, or a non-canonical
    text is reported; 'thu' checks the TRIAL_SEED_BASE seeds."""
    sets = sl.generate_sets(0)
    sha = hashlib.sha256(sl.canonical_text(sets).encode('utf-8')).hexdigest()
    rec, probs = sl.verify_screen_source(sets, sha)
    assert probs == [] and rec['khop_tung_bit'] and rec['khop_van_ban']
    assert rec['seed_base'] == sl.SEED_BASE
    assert rec['seeds'] == {s: sl.SEED_BASE + t for t, s in enumerate(sl.SETS)}
    bad = {s: np.array(a, copy=True) for s, a in sets.items()}          # one ulp moved in S3
    bad['S3'][0, 0] = np.nextafter(bad['S3'][0, 0], 1.0)
    rec, probs = sl.verify_screen_source(bad, sha)
    assert rec['tap_lech'] == ['S3'] and not rec['khop_tung_bit'] and len(probs) == 1
    bad = {s: np.array(a, copy=True) for s, a in sets.items()}          # S1 only
    bad['S1'][5, 3] = np.nextafter(bad['S1'][5, 3], 0.0)
    rec, _ = sl.verify_screen_source(bad, sha)
    assert rec['tap_lech'] == ['S1']
    small = {s: np.zeros((sl.SET_ROWS[s], 7)) for s in sl.SETS}         # -0.0 vs 0.0: same by ==

    def fake_gen(r=0, base=sl.SEED_BASE):
        out = {s: np.array(a, copy=True) for s, a in small.items()}
        out['S0'][0, 0] = -0.0
        return out
    monkeypatch.setattr(sl, 'generate_sets', fake_gen)
    rec, probs = sl.verify_screen_source(small, sha)
    assert 'S0' in rec['tap_lech'] and probs
    monkeypatch.undo()
    rec, probs = sl.verify_screen_source(sets, '0' * 64)                # right bits, wrong text
    assert rec['khop_tung_bit'] and not rec['khop_van_ban'] and len(probs) == 1
    assert 'text' in probs[0] and 'canonical' in probs[0]
    thu = sl.generate_sets(0, sl.TRIAL_SEED_BASE)                       # the tool-check family
    sha_thu = hashlib.sha256(sl.canonical_text(thu).encode('utf-8')).hexdigest()
    rec, probs = sl.verify_screen_source(thu, sha_thu, che_do='thu')
    assert probs == [] and rec['khop_tung_bit'] and rec['khop_van_ban']
    assert rec['seeds']['S0'] == sl.TRIAL_SEED_BASE
    rec, probs = sl.verify_screen_source(sets, sha, che_do='thu')       # main sets, thu seeds
    assert rec['tap_lech'] == list(sl.SETS)
    rec, probs = sl.verify_screen_source(thu, sha_thu)                  # thu sets, main seeds
    assert rec['tap_lech'] == list(sl.SETS)
    miss = {s: a for s, a in sets.items() if s != 'S2'}                 # a missing set
    rec, _ = sl.verify_screen_source(miss, sha)
    assert rec['tap_lech'] == ['S2']


def test_c3_36_source_seeds_validation():
    """C3-36: source_seeds refuses an unknown mode, a bad r (bool included) and 'thu' with
    r != 0; the seeds of ('chinh', 2) are SEED_BASE + 20 + t."""
    for bad in (('thu', 1), ('x', 0), ('chinh', -1), ('chinh', True)):
        with pytest.raises(ValueError):
            sl.source_seeds(*bad)
    base, seeds = sl.source_seeds('chinh', 2)
    assert base == sl.SEED_BASE
    assert seeds == {s: sl.SEED_BASE + 20 + t for t, s in enumerate(sl.SETS)}
    base, seeds = sl.source_seeds('thu', 0)
    assert base == sl.TRIAL_SEED_BASE and seeds['S0'] == sl.TRIAL_SEED_BASE


def test_c3_36_committed_files_bit_exact():
    """C3-36: the committed main file is generate_sets(0) bit for bit and canonical text; the
    committed tool-check file (make_s03_screen_u --thu) is the TRIAL_SEED_BASE family."""
    sets, info = sl.load_screen_u(sl.SCREEN_U, trial=True)
    gen = sl.generate_sets(0)
    assert hashlib.sha256(sl.canonical_text(gen).encode('utf-8')).hexdigest() == \
        info['sha256_lf']
    assert info['nguon_sinh']['khop_tung_bit'] and info['nguon_sinh']['khop_van_ban']
    sets, info = sl.load_screen_u(sl.SCREEN_U_THU, trial=True, che_do='thu')
    gen = sl.generate_sets(0, sl.TRIAL_SEED_BASE)
    assert hashlib.sha256(sl.canonical_text(gen).encode('utf-8')).hexdigest() == \
        info['sha256_lf']
    assert info['nguon_sinh']['khop_tung_bit'] and info['nguon_sinh']['khop_van_ban']
    assert info['nguon_sinh']['seeds']['S0'] == sl.TRIAL_SEED_BASE


# ── C3-48: a PID run without the clamp indicator stops the step ─────────────────

def test_c3_48_check_kep_rows():
    """C3-48: ADRC, attempt-2 and run_error rows are exempt; NaN fractions are values; a missing
    key, '' or None kep_frac or a non-empty kep_note of a first-attempt PID row stops S03."""
    good = dict(attempt=1, status='ok', cand_id=7, kep_frac_vx=0.5, kep_frac_vy=0.0,
                kep_frac_wz=float('nan'), kep_note='')
    sl.check_kep_rows('ADRC', [dict(attempt=1, status='ok')])            # ADRC exempt
    sl.check_kep_rows('PID', [good])                                     # NaN is a value
    sl.check_kep_rows('PID', [dict(good, attempt=2, kep_note='kep_lech set', kep_frac_vx='')])
    sl.check_kep_rows('PID', [dict(good, status='run_error', kep_note='', kep_frac_vx='')])
    for bad in (dict(good, kep_frac_vx=''), dict(good, kep_frac_vx=None),
                {k: v for k, v in good.items() if k != 'kep_frac_vx'},
                dict(good, kep_note='kep_lech set')):
        with pytest.raises(s3.S03Stop, match='7'):
            sl.check_kep_rows('PID', [bad])


# ── C3-39 / D2A-2: live phase entries ───────────────────────────────────────────

def test_c3_39_pha_notes_on_stop(tmp_path, monkeypatch):
    """C3-39 / D2A-2: a run_jobs stop inside S1 leaves .pha_notes on the exception: S1
    'dung_giua_pha' carrying the .n_luot counts of run_jobs, S0 'xong'."""
    def jobs(jobs_, work, n, lp):
        if any(j['name'].startswith('S1_') for j in jobs_):
            exc = s3.S03Stop('boom')
            exc.n_luot = (7, 2)
            raise exc
        return [sl.do_job(j) for j in jobs_]
    _patch(monkeypatch, _fake_gate())
    monkeypatch.setattr(sl, 'run_jobs', jobs)
    screen = _screen(40, 100, dict(S1=0.5, S2=0.5, S2b=0.5))
    W = _W()
    with pytest.raises(s3.S03Stop) as ei:
        sl.run_step(str(tmp_path), 'd.dll', s3.CONTROLLERS, [0.5], MU, W, _quet(W), screen, 2,
                    tool_check=dict(n0=40, n=100))
    pha = {p['pha']: p for p in ei.value.pha_notes}
    s1 = pha['S1']
    assert s1['n_luot_thuc'] == 7 and s1['n_chay_lai_chan_doan'] == 2
    assert s1['n_luot_tong'] == 9 and s1['trang_thai'] == 'dung_giua_pha'
    assert isinstance(s1['giay_may'], (int, float)) and isinstance(s1['moc_ket_thuc'], str)
    assert pha['S0']['trang_thai'] == 'xong' and 'S2' not in pha


def test_c3_39_s0_giay_may_is_t_s0(tmp_path, monkeypatch):
    """C3-39: giay_may of the S0 phase entry equals t_S0 of so_dem.json (the whole phase)."""
    res, W = _run(tmp_path, monkeypatch, _screen(40, 100, dict(S1=0.5, S2=0.5, S2b=0.5)),
                  _fake_gate())
    sd = json.load(open(tmp_path / 'sangloc_thu' / 'so_dem.json', encoding='utf-8'))
    pha = {p['pha']: p for p in res['pha']}
    assert pha['S0']['giay_may'] == sd['t_S0']
    for p in res['pha']:
        assert p['trang_thai'] == 'xong' and p['n_luot_tong'] == \
            p['n_luot_thuc'] + p['n_chay_lai_chan_doan']
        assert isinstance(p['moc_bat_dau'], str) and isinstance(p['moc_ket_thuc'], str)


# ── D2A-3: sealed/sangloc_thu/loi_S0.json ────────────────────────────────────────

def test_d2a_3_loi_s0(tmp_path, monkeypatch):
    """D2A-3: loi_S0.json is written always when S0 ended — the keys of LOI_S0_KEYS /
    LOI_LUOT_KEYS only, the unclassifiable counts of so_dem.json, one 'lan_1' entry per
    attempt-1 run_error of evals_S0.csv and a 'chan_doan' entry per attempt-2 row; a run
    without errors has zero counts and an empty list."""
    u_s0 = np.full((40, 7), 0.5)
    u_s0[:3] = 0.9                                   # pos_kp ~91: run_error rows 1-3
    _run(tmp_path, monkeypatch,
         dict(S0=u_s0, S1=np.full((100, 7), 0.5), S2=np.full((100, 7), 0.5),
              S2b=np.full((100, 7), 0.5)),
         _fake_gate(bad=lambda g, v, mu: 60.0 < g['pos_kp'] < 100.0),
         tool=dict(n0=40, n=100))
    doc = json.load(open(tmp_path / 'sangloc_thu' / 'loi_S0.json', encoding='utf-8'))
    assert list(doc) == list(sl.LOI_S0_KEYS)
    sd = json.load(open(tmp_path / 'sangloc_thu' / 'so_dem.json', encoding='utf-8'))
    assert set(doc['n_bo_khong_phan_loai']) == set(s3.CONTROLLERS)
    for ctrl in s3.CONTROLLERS:
        assert set(doc['n_bo_khong_phan_loai'][ctrl]) == set(sl.MODES)
        for m in sl.MODES:
            assert doc['n_bo_khong_phan_loai'][ctrl][m] == \
                sd['dem'][ctrl]['che_do'][m]['n_khong_phan_loai']
    for e in doc['luot_loi']:
        assert set(e) == set(sl.LOI_LUOT_KEYS) and e['pha'] in ('lan_1', 'chan_doan')
    ev = _read_csv(tmp_path / 'sangloc_thu' / 'evals_S0.csv')
    n_lan1 = sum(r['attempt'] == '1' and r['status'] == 'run_error' for r in ev)
    n_cd = sum(r['attempt'] == '2' for r in ev)
    assert sum(e['pha'] == 'lan_1' for e in doc['luot_loi']) == n_lan1 > 0
    assert sum(e['pha'] == 'chan_doan' for e in doc['luot_loi']) == n_cd
    txt = json.dumps(doc)
    for word in ('e_perp', 'omega', 'kp_xy', 'pos_kp', '0.9'):
        assert word not in txt
    _run(tmp_path / 'b', monkeypatch, _screen(40, 100, dict(S1=0.5, S2=0.5, S2b=0.5)),
         _fake_gate())
    doc = json.load(open(tmp_path / 'b' / 'sangloc_thu' / 'loi_S0.json', encoding='utf-8'))
    assert doc['luot_loi'] == []
    assert all(v == 0 for c in doc['n_bo_khong_phan_loai'].values() for v in c.values())


# ── C3-40 / D2A-4: worker death, the diagnosis rerun and the wait limits ────────

def _fake_worker(tmp_path):
    """A worker-process script (C3-40 tests): the real sl.job_main with a fake gate and
    firmware; environment variables make it os._exit(9) at a gain set (S03_TEST_KILL=<k_psi
    repr>, matched when abs(g['k_psi'] / k - 1) < 1e-12 on a PID run), sleep
    (S03_TEST_SLEEP=<k_psi>:<seconds>), raise S03Stop (S03_TEST_STOP=<k_psi>) or exit before
    job_main (S03_TEST_EXIT_AT_START=1)."""
    script = tmp_path / 'fake_worker.py'
    fwsim_dir = os.path.abspath(os.path.join(HERE, '..'))
    script.write_text(textwrap.dedent('''\
        import os, sys, time
        sys.path.insert(0, {fwsim!r})
        import s03_doe as s3
        import s03_sangloc as sl
        import s04_ga as s4

        KILL = os.environ.get('S03_TEST_KILL')
        SLEEP = os.environ.get('S03_TEST_SLEEP')
        STOP = os.environ.get('S03_TEST_STOP')

        def _match(k_psi_text, g):
            return k_psi_text is not None and abs(g['k_psi'] / float(k_psi_text) - 1) < 1e-12

        def gate(fw, path, ctrl, g, v, mu):
            if ctrl == 'PID':
                if _match(KILL, g):
                    os._exit(9)
                if SLEEP is not None:
                    k_, s_ = SLEEP.split(':')
                    if _match(k_, g):
                        time.sleep(float(s_))
                if _match(STOP, g):
                    raise s3.S03Stop('fake tool stop')
            kep = dict(kep_frac_vx=0.0, kep_frac_vy=0.0, kep_frac_wz=0.0, kep_note='') \\
                if ctrl == 'PID' else \\
                dict(kep_frac_vx='', kep_frac_vy='', kep_frac_wz='', kep_note='')
            return dict(status='ok', completed='true', e_perp_max=0.05, tv_duty=0.4,
                        sat_frac=0.0, t_end=1.0, e_theta_lane_max_deg=0.5,
                        e_theta_cmd_max_deg=0.5, ds_absmax=1.0, e_trk_max=0.05,
                        e_bao_max=0.02, **kep), {{'t': [0.0]}}

        s4.safe_run_gate = gate
        sl._firmware = lambda dll: object()
        if os.environ.get('S03_TEST_EXIT_AT_START'):
            os._exit(5)
        _p = sys.argv[sys.argv.index('--job') + 1]
        print('worker job file', os.path.basename(_p), flush=True)
        sl.job_main(_p)
        '''.format(fwsim=fwsim_dir)), encoding='utf-8')
    return str(script)


def _popen_fake_worker(monkeypatch, script, fail_at=None, procs=None):
    """subprocess.Popen: real for every command without '--job'; the n-th '--job' spawn raises
    OSError('FakeSpawnError') when n == fail_at, else runs the fake-worker script with the same
    '--job <path>' arguments and appends the process to procs (works on the original code too)."""
    real = subprocess.Popen
    count = [0]

    def popen(cmd, *a, **k):
        if '--job' not in list(cmd):
            return real(cmd, *a, **k)
        count[0] += 1
        if count[0] == fail_at:
            raise OSError('FakeSpawnError')
        p = real([cmd[0], script] + list(cmd[2:]), *a, **k)
        if procs is not None:
            procs.append(p)
        return p
    monkeypatch.setattr(subprocess, 'Popen', popen)
    return count


def _job_pid(name, work, rows, speeds=(0.5,), mu=MU):
    return dict(name=name, dll='fake.dll', controller='PID', speeds=list(speeds),
                mu_levels=list(mu), rows=rows, work=os.path.join(str(work), name))


def _five_rows(n=5):
    return [(i + 1, dict(FAKE_DEFAULTS['PID'], k_psi=1.0 + i)) for i in range(n)]


@pytest.mark.parametrize('k', [1, 3, 5])
def test_c3_40_worker_dies_mid_set(tmp_path, monkeypatch, k):
    """C3-40 / D2A-4: a worker that dies while running gain set k: the sets it reported 'xong'
    are kept, set k is a run_error (the death in 'error') with ONE diagnosis rerun in a new
    process (attempt 2, never replacing), the rest of the job runs in a '<name>_tiep1' worker;
    run_jobs returns, no worker is alive, no job file is left."""
    script = _fake_worker(tmp_path)
    work = str(tmp_path / 'wk')
    log_path = str(tmp_path / 'tien_trinh.log')
    procs = []
    _popen_fake_worker(monkeypatch, script, procs=procs)
    monkeypatch.setenv('S03_TEST_KILL', repr(1.0 + (k - 1)))
    calls = []

    def diag(dll, runs, w, timeout=None):
        calls.append((len(runs), timeout))
        return _fake_new_process(dll, runs, w, timeout=timeout)
    monkeypatch.setattr(s3, 'run_in_new_process', diag)
    jobs = [_job_pid('J', work, _five_rows())]
    res = sl.run_jobs(jobs, work, 2, log_path)
    assert len(procs) == (1 if k == 5 else 2)
    assert calls == [(1, sl.DIAG_WAIT_MAX_S)]            # ONE diagnosis of the 1 speed at mu_mid
    by_row = {rn: rows for rn, rows in res[0]}
    for rn, rows in by_row.items():
        if rn != k:
            assert [e['status'] for e in rows] == ['ok'] * 3
            assert all(e['attempt'] == 1 for e in rows)
        else:
            assert len(rows) == 2
            assert rows[0]['attempt'] == 1 and rows[0]['status'] == 'run_error'
            assert 'worker process died (exit code 9)' in rows[0]['error']
            assert rows[1]['attempt'] == 2
            sc = sl.score_modes(rows, 'ok', [0.5], MU)
            assert all(not sc[m]['phan_loai_duoc'] for m in sl.MODES)
    assert not os.path.exists(work) or not os.listdir(work)
    log = open(log_path, encoding='utf-8', errors='replace').read()
    assert 'job J' in log and ('job J_tiep1' in log) == (k < 5)


def test_c3_40_diagnosis_fails(tmp_path, monkeypatch):
    """C3-40: when the diagnosis rerun itself raises, the attempt-2 row of every speed is
    'diagnosis_failed' with the exception text; the attempt-1 run_error rows stay."""
    script = _fake_worker(tmp_path)
    work = str(tmp_path / 'wk')
    log_path = str(tmp_path / 'tien_trinh.log')
    _popen_fake_worker(monkeypatch, script)
    monkeypatch.setenv('S03_TEST_KILL', repr(2.0))
    monkeypatch.setattr(s3, 'run_in_new_process',
                        lambda *a, **k: (_ for _ in ()).throw(s3.S03Stop('child process failed (9)')))
    jobs = [_job_pid('J', work, _five_rows())]
    res = sl.run_jobs(jobs, work, 2, log_path)
    rows = dict(res[0])[2]
    assert rows[0]['status'] == 'run_error' and 'worker process died' in rows[0]['error']
    assert rows[1]['attempt'] == 2 and rows[1]['status'] == 'diagnosis_failed'
    assert 'child process failed (9)' in rows[1]['error']


def test_c3_40_exit_at_start_is_a_tool_error(tmp_path, monkeypatch):
    """C3-40: a worker dead before any 'bat_dau' (os._exit at start) is the existing tool
    error, not a run_error set."""
    script = _fake_worker(tmp_path)
    work = str(tmp_path / 'wk')
    log_path = str(tmp_path / 'tien_trinh.log')
    _popen_fake_worker(monkeypatch, script)
    monkeypatch.setenv('S03_TEST_EXIT_AT_START', '1')
    jobs = [_job_pid('J', work, _five_rows())]
    with pytest.raises(s3.S03Stop, match='worker'):
        sl.run_jobs(jobs, work, 2, log_path)
    assert not os.path.exists(work) or not os.listdir(work)


def test_c3_40_wait_limit(tmp_path, monkeypatch):
    """C3-40: a worker over JOB_WAIT_MAX_S is killed (the set it was running is a run_error
    with the wait-limit text), the rest runs in a continuation; the call returns far below
    the sleep length."""
    script = _fake_worker(tmp_path)
    work = str(tmp_path / 'wk')
    log_path = str(tmp_path / 'tien_trinh.log')
    procs = []
    _popen_fake_worker(monkeypatch, script, procs=procs)
    monkeypatch.setenv('S03_TEST_SLEEP', f'{2.0!r}:40')
    monkeypatch.setattr(sl, 'JOB_WAIT_MAX_S', 3.0, raising=False)
    monkeypatch.setattr(s3, 'run_in_new_process', _fake_new_process)
    jobs = [_job_pid('J', work, _five_rows(3))]
    t0 = time.monotonic()
    res = sl.run_jobs(jobs, work, 2, log_path)
    assert time.monotonic() - t0 < 30
    by_row = dict(res[0])
    rows = by_row[2]
    assert rows[0]['status'] == 'run_error' and 'wait limit' in rows[0]['error']
    assert [e['status'] for e in by_row[1] + by_row[3]] == ['ok'] * 6
    assert not os.path.exists(work) or not os.listdir(work)
    for p in procs:
        if p.poll() is None:                                   # nothing of run_jobs stays alive
            p.kill()


def test_c3_40_spawn_error_reaps_the_running_worker(tmp_path, monkeypatch):
    """C3-40: a spawn failure (OSError) propagates but the running worker is killed and reaped
    and no job file is left under work."""
    script = _fake_worker(tmp_path)
    work = str(tmp_path / 'wk')
    log_path = str(tmp_path / 'tien_trinh.log')
    procs = []
    _popen_fake_worker(monkeypatch, script, fail_at=2, procs=procs)
    monkeypatch.setenv('S03_TEST_SLEEP', f'{1.0!r}:60')
    jobs = [_job_pid('A', work, _five_rows(3)), _job_pid('B', work, _five_rows(3))]
    try:
        with pytest.raises(OSError, match='FakeSpawnError'):
            sl.run_jobs(jobs, work, 2, log_path)
        assert procs[0].poll() is not None                     # reaped before run_jobs returned
        assert not os.path.exists(work) or not os.listdir(work)
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()


def test_c3_40_stop_carries_n_luot(tmp_path, monkeypatch):
    """C3-40: a worker reporting ok=False raises its S03Stop carrying .n_luot — the finished
    sets' attempt-1 / attempt-2 rows (here 2 sets x 3 attempt-1 runs = 6)."""
    script = _fake_worker(tmp_path)
    work = str(tmp_path / 'wk')
    log_path = str(tmp_path / 'tien_trinh.log')
    _popen_fake_worker(monkeypatch, script)
    monkeypatch.setenv('S03_TEST_STOP', repr(3.0))
    monkeypatch.setattr(s3, 'run_in_new_process', _fake_new_process)
    jobs = [_job_pid('J', work, _five_rows())]
    with pytest.raises(s3.S03Stop, match='fake tool stop') as ei:
        sl.run_jobs(jobs, work, 2, log_path)
    assert ei.value.n_luot[0] == 6


def test_c3_40_n_luot_not_double_counted(tmp_path, monkeypatch):
    """C3-40: the 'xong' sets of a worker that died while running a set are merged into the
    result AND are not counted again in exc.n_luot when a later stop goes up: the first
    worker finishes row 1 and dies in row 2, the continuation finishes row 3 and reports
    the tool stop in row 4 -> n_luot == (6, 0) (with the double count it is 9)."""
    script = _fake_worker(tmp_path)
    work = str(tmp_path / 'wk')
    log_path = str(tmp_path / 'tien_trinh.log')
    _popen_fake_worker(monkeypatch, script)
    monkeypatch.setenv('S03_TEST_KILL', repr(2.0))
    monkeypatch.setenv('S03_TEST_STOP', repr(4.0))
    monkeypatch.setattr(s3, 'run_in_new_process', _fake_new_process)
    jobs = [_job_pid('J', work, _five_rows())]
    with pytest.raises(s3.S03Stop, match='fake tool stop') as ei:
        sl.run_jobs(jobs, work, 2, log_path)
    assert ei.value.n_luot == (6, 0)
    assert not os.path.exists(work) or not os.listdir(work)


def test_c3_40_read_part_cut_record(tmp_path):
    """C3-40: _read_part keeps the good records and stops at a record cut by the death of the
    worker: a truncated pickle, a garbage tail and a complete pickle that is not a 3-tuple
    each leave (done, started) = ([(1, rows)], 2); a missing file gives ([], None)."""
    import pickle as _pkl
    rows = [dict(attempt=1, status='ok')]
    p = str(tmp_path / 'job_X')

    def build(tail):
        with open(p + '.part', 'wb') as f:
            _pkl.dump(('bat_dau', 1, None), f)
            _pkl.dump(('xong', 1, rows), f)
            _pkl.dump(('bat_dau', 2, None), f)
            f.write(tail)

    rec = _pkl.dumps(('xong', 2, rows))
    build(rec[:len(rec) // 2])                                     # a truncated pickle
    assert sl._read_part(p) == ([(1, rows)], 2)
    build(b'\x80\x05garbage')                                      # a garbage tail
    assert sl._read_part(p) == ([(1, rows)], 2)
    build(_pkl.dumps('x'))                                         # not a 3-tuple
    assert sl._read_part(p) == ([(1, rows)], 2)
    assert sl._read_part(str(tmp_path / 'none')) == ([], None)
