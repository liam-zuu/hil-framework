"""T9: heading-loop latch at the run_active edge, three test cases on the firmware's own C code
(fwsim/t9_heading_cases.py; J v6 section 5 command-path condition (1), section 7).
Checks the log signature of each case row by row, and that the reference reading ref_cond1()
marks the run bad (control case: not bad). Needs fwsim/build/fwsim.dll with the position loop.
Run: python -m pytest fwsim/tests/test_t9_heading.py"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import fwsim  # noqa: E402
import t9_heading_cases as T  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.isfile(os.path.join(fwsim.HERE, 'build', 'fwsim.dll')),
                                reason='fwsim.dll not built')

AGE_MAX = 60.0     # test threshold only (J locks the real one at C8)


@pytest.fixture(scope='module')
def runs():
    fw = fwsim.FirmwareSim()
    if fw.pos_cols == 0:
        pytest.skip('fwsim.dll without the position loop')
    return {c.name: T.as_logged(T.simulate(fw, c)) for c in T.cases()}


def edge_index(rows):
    return next(i for i in range(1, len(rows)) if rows[i]['run_active'] == 1 and rows[i - 1]['run_active'] == 0)


def window(rows):
    """Edge to the last run_active row (the run; rows after the run ends are not part of it)."""
    i = edge_index(rows)
    j = max(n for n in range(len(rows)) if rows[n]['run_active'] == 1)
    return rows[i:j + 1]


def test_cases_are_the_three_plus_control():
    names = [c.name for c in T.cases()]
    assert names[0] == 'ok'
    assert {n.split('_')[0] + '_' + n.split('_')[1] for n in names[1:]} == {
        'mat_yaw', 'chot_muon', 'reset_giua', 'canh_doi'}


def test_ok_latches_on_the_edge_tick(runs):
    r = runs['ok']
    i = edge_index(r)
    assert r[i]['timestamp_us'] == T.EDGE_US
    pre, e = r[i - 1], r[i]
    assert (pre['run_active'], pre['head_on']) == (0, 0)
    assert (e['run_active'], e['r_valid'], e['head_on'], e['psi_ref'], e['e_psi']) == (1, 1, 1, 0.0, 0.0)
    # same tick: the heading loop ran with this row's r_valid and the yaw it latched is psi0
    assert e['psi_reset_count'] == 0 and e['pos_reset_n'] == 1
    # after the edge psi_ref integrates wz_ref (0.3 rad/s), counters stay
    assert r[i + 10]['psi_ref'] == pytest.approx(0.3 * 0.10, abs=1e-4)
    win = window(r)
    assert {x['psi_reset_count'] for x in win} == {0}
    assert {x['reset_n'] for x in win} == {0}
    assert {x['pos_reset_n'] for x in win} == {1}
    assert all(x['head_on'] == 1 for x in win)
    bad, why, info = T.ref_cond1(r, AGE_MAX)
    assert not bad, why
    assert info['gap_ms'] == 10.0
    assert info['age_bound_ms'] == pytest.approx(info['age_r_ms'] + 0.05 + 10.0)


def test_mat_yaw_edge_not_latched_and_never_on(runs):
    r = runs['mat_yaw']
    i = edge_index(r)
    e = r[i]
    assert (e['run_active'], e['r_valid'], e['head_on']) == (1, 0, 0)
    assert all(x['head_on'] == 0 for x in window(r))
    assert {x['psi_reset_count'] for x in window(r)} == {0}      # nothing but the edge row shows it
    bad, why, _ = T.ref_cond1(r, AGE_MAX)
    assert bad and 'edge_not_latched' in why and 'age_bound_undetermined' in why


def test_chot_muon_a_late_latch_only_the_edge_row_shows_it(runs):
    r = runs['chot_muon_a']
    i = edge_index(r)
    assert (r[i]['r_valid'], r[i]['head_on']) == (0, 0)
    j = next(k for k in range(i, len(r)) if r[k]['head_on'] == 1)
    # first RV report after the drop window (EDGE + 300 ms) lands on the next tick: latched there
    assert 300_000 <= r[j]['timestamp_us'] - T.EDGE_US <= 300_000 + T.RV_PERIOD_US
    assert (r[j]['r_valid'], r[j]['psi_ref'], r[j]['e_psi']) == (1, 0.0, 0.0)
    assert r[j - 1]['r_valid'] == 0
    # late latch moves no counter: psi_reset_count, reset_n, pos_reset_n constant over the run
    for c in ('psi_reset_count', 'reset_n', 'pos_reset_n'):
        assert len({x[c] for x in window(r)}) == 1, c
    bad, why, _ = T.ref_cond1(r, AGE_MAX)
    assert bad and why[0] == 'edge_not_latched'


def test_chot_muon_b_stale_but_valid_yaw_caught_by_the_age_bound(runs):
    r = runs['chot_muon_b']
    i = edge_index(r)
    e = r[i]
    assert (e['r_valid'], e['head_on'], e['psi_ref']) == (1, 1, 0.0)    # the firmware latches it
    assert 90.0 <= e['age_r_ms'] < 125.0                                # under t_age = 2.5 x 50 ms
    bad, why, info = T.ref_cond1(r, AGE_MAX)
    assert bad and why == ['age_bound_over']
    assert info['age_bound_ms'] == pytest.approx(e['age_r_ms'] + 10.05)
    assert not T.ref_cond1(r, 200.0)[0]                                 # the threshold decides it


def test_reset_giua_a_chip_reset(runs):
    r = runs['reset_giua_a']
    i = edge_index(r)
    k = next(n for n in range(i, len(r)) if r[n]['reset_n'] == 1)
    assert r[k]['timestamp_us'] - T.EDGE_US == 1_000_000
    # the reset tick: loop off, psi_reset_count + 1 in the same row as reset_n + 1
    assert (r[k]['head_on'], r[k]['psi_reset_count']) == (0, 1)
    assert r[k - 1]['psi_reset_count'] == 0
    # back on after t_settle (500 ms) with a new psi0
    j = next(n for n in range(k, len(r)) if r[n]['head_on'] == 1)
    assert 500_000 <= r[j]['timestamp_us'] - r[k]['timestamp_us'] <= 500_000 + T.RV_PERIOD_US + T.TICK_US
    assert r[j]['psi_ref'] == 0.0
    bad, why, _ = T.ref_cond1(r, AGE_MAX)
    assert bad and set(why) == {'psi_reset_count_changed', 'reset_n_changed'}


def test_reset_giua_b_yaw_back_with_large_error(runs):
    r = runs['reset_giua_b']
    i = edge_index(r)
    k = next(n for n in range(i, len(r)) if r[n]['psi_reset_count'] == 1)
    # yaw back after 1 s of loss: psi_ref had integrated 0.3 rad/s more than psi_imu (> 0.35 rad)
    assert r[k]['r_valid'] == 1 and r[k - 1]['r_valid'] == 0 and r[k]['head_on'] == 1
    assert r[k - 1]['psi_ref'] > 0.35
    assert r[k]['psi_ref'] == 0.0 and r[k]['e_psi'] == 0.0        # psi_ref := psi_imu (car still: 0)
    assert {x['reset_n'] for x in window(r)} == {0} and {x['pos_reset_n'] for x in window(r)} == {1}
    bad, why, _ = T.ref_cond1(r, AGE_MAX)
    assert bad and why == ['psi_reset_count_changed']


def test_reset_giua_c_run_active_drop(runs):
    r = runs['reset_giua_c']
    i = edge_index(r)
    drop = next(n for n in range(i, len(r)) if r[n]['run_active'] == 0)
    back = next(n for n in range(drop, len(r)) if r[n]['run_active'] == 1)
    assert r[drop]['head_on'] == 0
    assert r[back]['pos_reset_n'] == r[i]['pos_reset_n'] + 1           # pose reset on the re-edge
    assert (r[back]['head_on'], r[back]['psi_ref']) == (1, 0.0)        # new psi0, ...
    assert r[back]['psi_reset_count'] == 0                              # ... psi_reset_count does not move
    bad, why, _ = T.ref_cond1(r, AGE_MAX)
    assert bad and why == ['pos_reset_n_changed']


def test_age_bound_needs_the_next_row():
    row = dict(timestamp_us=0, log_seq=10, run_active=1, r_valid=1, head_on=1, psi_ref=0.0, age_r_ms=5.0,
               psi_reset_count=0, reset_n=0, pos_reset_n=1)
    pre = dict(row, log_seq=9, run_active=0, head_on=0)
    nxt = dict(row, log_seq=11, timestamp_us=12_000)
    assert T.ref_cond1([pre, row, nxt], AGE_MAX)[2]['age_bound_ms'] == pytest.approx(17.05)
    assert T.ref_cond1([pre, row, dict(nxt, log_seq=12)], AGE_MAX)[1] == ['age_bound_undetermined']
    assert T.ref_cond1([pre, row], AGE_MAX)[1] == ['age_bound_undetermined']
    assert T.ref_cond1([pre, dict(row, age_r_ms=float('nan')), nxt], AGE_MAX)[1] == ['age_bound_undetermined']


def test_canh_doi_mode_yaw_valid_but_not_latched_on_the_edge(runs):
    r = runs['canh_doi_mode']
    i = edge_index(r)
    e = r[i]
    # reset_event (mode change) wins over the start of the run: loop off, psi_ref still the pre-run
    # value (psi_ref tracks psi_imu while not running), r_valid = 1
    assert (e['run_active'], e['r_valid'], e['head_on']) == (1, 1, 0)
    assert e['psi_ref'] != 0.0
    assert (r[i + 1]['head_on'], r[i + 1]['psi_ref']) == (1, 0.0)       # latched one tick late
    for c in ('psi_reset_count', 'reset_n', 'pos_reset_n'):
        assert len({x[c] for x in window(r)}) == 1, c                  # no counter shows it
    bad, why, _ = T.ref_cond1(r, AGE_MAX)
    assert bad and why == ['edge_not_latched']


@pytest.mark.parametrize('field,value', [('r_valid', 0), ('head_on', 0), ('psi_ref', 0.0001)])
def test_each_edge_condition_alone_marks_the_run(field, value):
    row = dict(timestamp_us=0, log_seq=10, run_active=1, r_valid=1, head_on=1, psi_ref=0.0, age_r_ms=5.0,
               psi_reset_count=0, reset_n=0, pos_reset_n=1)
    pre = dict(row, log_seq=9, run_active=0, head_on=0)
    nxt = dict(row, log_seq=11, timestamp_us=10_000)
    assert not T.ref_cond1([pre, row, nxt], AGE_MAX)[0]
    bad, why, _ = T.ref_cond1([pre, dict(row, **{field: value}), nxt], AGE_MAX)
    assert bad and 'edge_not_latched' in why
