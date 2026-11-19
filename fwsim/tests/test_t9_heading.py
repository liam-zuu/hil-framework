"""T9: heading-loop latch at the run_active edge, three test cases on the firmware's own C code
(fwsim/t9_heading_cases.py; J v6 section 5 command-path condition (1), section 7).
Checks the log signature of each case row by row, and that the reference reading ref_cond1()
marks the run bad (control case: not bad). The rows are built from the DLL's controller outputs
(fw_sim.c does not compile main.c / logger.c), see the module docstring.
Needs fwsim/build/fwsim.dll with the position loop.
Run: python -m pytest fwsim/tests/test_t9_heading.py"""
import math
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
def fw():
    f = fwsim.FirmwareSim()
    if f.pos_cols == 0:
        pytest.skip('fwsim.dll without the position loop')
    return f


@pytest.fixture(scope='module')
def raw(fw):
    return {c.name: T.simulate(fw, c) for c in T.cases()}


@pytest.fixture(scope='module')
def runs(raw):
    return {k: T.as_logged(v) for k, v in raw.items()}


def edge_index(rows):
    return next(i for i in range(1, len(rows)) if rows[i]['run_active'] == 1 and rows[i - 1]['run_active'] == 0)


def window(rows):
    """Edge to the last run_active row (the run; rows after the run ends are not part of it)."""
    i = edge_index(rows)
    j = max(n for n in range(len(rows)) if rows[n]['run_active'] == 1)
    return rows[i:j + 1]


def row(**kw):
    """A synthetic edge row that passes every check (age 5.0 ms)."""
    base = dict(timestamp_us=0, log_seq=10, run_active=1, r_valid=1, rv_n=7, head_on=1, psi_ref=0.0,
                age_r_ms=5.0, psi_reset_count=0, reset_n=0, pos_reset_n=1)
    base.update(kw)
    return base


def trio(edge=None, nxt=None, pre=None):
    e = edge or row()
    return [pre or dict(row(), log_seq=9, run_active=0, head_on=0, timestamp_us=-10_000),
            e, nxt or dict(row(), log_seq=11, timestamp_us=10_000)]


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
    assert info['age_bound_ms'] == pytest.approx(info['age_r_ms'] + 0.05 + 10.0 + 0.003)


def test_psi0_is_the_report_held_on_the_edge_tick(raw):
    """psi0 = yaw of the latest report on the edge tick (report n carries YAW0 + n * step): the next
    report moves psi_imu by exactly one step; a psi0 from an older report would give two or more."""
    r = raw['ok']
    i = edge_index(r)
    n0 = r[i]['rv_n']
    j = next(k for k in range(i + 1, len(r)) if r[k]['rv_n'] == n0 + 1)
    assert all(r[k]['_psi_imu'] == 0.0 for k in range(i, j))
    assert r[j]['_psi_imu'] == pytest.approx(math.radians(T.YAW_STEP_DEG), rel=1e-3)


def test_mat_yaw_edge_not_latched_and_never_on(runs):
    r = runs['mat_yaw']
    i = edge_index(r)
    e = r[i]
    assert (e['run_active'], e['r_valid'], e['head_on']) == (1, 0, 0)
    assert all(x['head_on'] == 0 for x in window(r))
    assert {x['psi_reset_count'] for x in window(r)} == {0}      # nothing but the edge row shows it
    bad, why, info = T.ref_cond1(r, AGE_MAX)
    assert bad and 'edge_not_latched' in why and 'age_bound_undetermined' in why
    assert info['loai'] == 'duong_lenh'


def test_chot_muon_a_late_latch_only_the_edge_row_shows_it(raw, runs):
    r = runs['chot_muon_a']
    i = edge_index(r)
    assert (r[i]['r_valid'], r[i]['head_on']) == (0, 0)
    j = next(k for k in range(i, len(r)) if r[k]['head_on'] == 1)
    # first report after the drop window: EDGE + 303.7 ms, applied on the tick at + 310 ms, latched there
    assert r[j]['timestamp_us'] - T.EDGE_US == 310_000
    assert (r[j]['r_valid'], r[j]['psi_ref'], r[j]['e_psi']) == (1, 0.0, 0.0)
    assert r[j - 1]['r_valid'] == 0
    for c in ('psi_reset_count', 'reset_n', 'pos_reset_n'):
        assert len({x[c] for x in window(r)}) == 1, c
    bad, why, _ = T.ref_cond1(r, AGE_MAX)
    assert bad and why[0] == 'edge_not_latched'


def test_chot_muon_b_stale_but_valid_yaw_caught_by_the_age_bound(runs):
    r = runs['chot_muon_b']
    i = edge_index(r)
    e = r[i]
    assert (e['r_valid'], e['head_on'], e['psi_ref']) == (1, 1, 0.0)    # the firmware latches it
    assert e['age_r_ms'] == 96.3                                        # under t_age = 2.5 x 50 ms
    bad, why, info = T.ref_cond1(r, AGE_MAX)
    assert bad and why == ['age_bound_over']
    assert info['age_bound_ms'] == pytest.approx(96.3 + 0.05 + 10.0 + 0.003)
    assert not T.ref_cond1(r, 200.0)[0]                                 # the threshold decides it


def test_reset_giua_a_chip_reset(runs):
    r = runs['reset_giua_a']
    i = edge_index(r)
    k = next(n for n in range(i, len(r)) if r[n]['reset_n'] == 1)
    assert r[k]['timestamp_us'] - T.EDGE_US == 1_000_000
    assert (r[k]['head_on'], r[k]['psi_reset_count']) == (0, 1)        # same row as reset_n + 1
    assert r[k - 1]['psi_reset_count'] == 0
    j = next(n for n in range(k, len(r)) if r[n]['head_on'] == 1)      # back after t_settle 500 ms
    assert 500_000 <= r[j]['timestamp_us'] - r[k]['timestamp_us'] <= 500_000 + T.RV_PERIOD_US + T.TICK_US
    assert r[j]['psi_ref'] == 0.0
    bad, why, _ = T.ref_cond1(r, AGE_MAX)
    assert bad and set(why) == {'psi_reset_count_changed', 'reset_n_changed'}


def test_reset_giua_b_yaw_back_with_large_error(runs):
    r = runs['reset_giua_b']
    i = edge_index(r)
    lost = next(n for n in range(i, len(r)) if r[n]['r_valid'] == 0)
    assert r[lost]['timestamp_us'] - T.EDGE_US == 1_080_000
    k = next(n for n in range(i, len(r)) if r[n]['psi_reset_count'] == 1)
    assert r[k]['timestamp_us'] - T.EDGE_US == 2_010_000
    assert r[k]['r_valid'] == 1 and r[k - 1]['r_valid'] == 0 and r[k]['head_on'] == 1
    assert r[k - 1]['psi_ref'] == pytest.approx(0.6, abs=0.002)        # > e_psi_reset 0.35 rad
    assert abs(r[k]['e_psi']) == 0.0                                    # psi_ref := psi_imu
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


def test_canh_doi_mode_injected_mode_event_on_the_edge(runs):
    r = runs['canh_doi_mode']
    i = edge_index(r)
    e = r[i]
    # reset_event wins over the start of the run: loop off, psi_ref still the pre-run value
    assert (e['run_active'], e['r_valid'], e['head_on']) == (1, 1, 0)
    assert e['psi_ref'] != 0.0
    assert (r[i + 1]['head_on'], r[i + 1]['psi_ref']) == (1, 0.0)       # latched one tick late
    for c in ('psi_reset_count', 'reset_n', 'pos_reset_n'):
        assert len({x[c] for x in window(r)}) == 1, c
    bad, why, _ = T.ref_cond1(r, AGE_MAX)
    assert bad and why == ['edge_not_latched', 'age_bound_undetermined']


# ---- ref_cond1 on synthetic rows (Codex T9-1 .. T9-5) ----

def test_synthetic_good_run_passes():
    bad, why, info = T.ref_cond1(trio(), AGE_MAX)
    assert not bad, why
    assert info['age_bound_ms'] == pytest.approx(5.0 + 0.05 + 10.0 + 0.003)


@pytest.mark.parametrize('field,value', [('r_valid', 0), ('head_on', 0), ('psi_ref', 0.0001), ('rv_n', 0)])
def test_each_edge_condition_alone_marks_the_run(field, value):
    bad, why, _ = T.ref_cond1(trio(edge=row(**{field: value})), AGE_MAX)
    assert bad and 'edge_not_latched' in why


def test_missing_row_before_the_edge_is_a_measurement_failure():
    """T9-1: the 0 -> 1 change seen across a gap in log_seq is not edge evidence."""
    bad, why, info = T.ref_cond1(trio(pre=dict(row(), log_seq=8, run_active=0, head_on=0)), AGE_MAX)
    assert bad and why == ['thieu_dong_truoc_canh'] and info['loai'] == 'he_do'


def test_window_covers_run_active_zero_stretch_and_end_anchor():
    """T9-2: a counter change on a run_active = 0 row before the end anchor is inside the run."""
    rows = trio() + [dict(row(), log_seq=12, timestamp_us=20_000, run_active=0, reset_n=1),
                     dict(row(), log_seq=13, timestamp_us=30_000, run_active=0, reset_n=1)]
    assert T.ref_cond1(rows, AGE_MAX)[1] == []                           # default window: last active row
    bad, why, _ = T.ref_cond1(rows, AGE_MAX, end_seq=13)
    assert bad and why == ['reset_n_changed']


@pytest.mark.parametrize('nxt', [dict(row(), log_seq=12, timestamp_us=10_000),     # not log_seq + 1
                                 dict(row(), log_seq=11, timestamp_us=0),          # not later (T9-3)
                                 dict(row(), log_seq=11, timestamp_us=-5_000)])
def test_age_bound_needs_a_later_next_row(nxt):
    rows = trio(nxt=nxt)
    assert T.ref_cond1(rows, AGE_MAX)[1] == ['age_bound_undetermined']
    assert T.ref_cond1(rows[:2], AGE_MAX)[1] == ['age_bound_undetermined']


@pytest.mark.parametrize('edge', [{k: v for k, v in row().items() if k != 'r_valid'},
                                  {k: v for k, v in row().items() if k != 'age_r_ms'},
                                  row(age_r_ms=float('nan')), row(age_r_ms=''), row(rv_n='')])
def test_missing_or_bad_fields_never_pass(edge):
    """T9-4: missing flag or age, NaN, empty field: bad, no exception."""
    bad, why, _ = T.ref_cond1(trio(edge=edge), AGE_MAX)
    assert bad and 'age_bound_undetermined' in why


def test_negative_logged_age_with_valid_flags_is_normal():
    bad, why, info = T.ref_cond1(trio(edge=row(age_r_ms=-1.2)), AGE_MAX)
    assert not bad and info['age_bound_ms'] == pytest.approx(-1.2 + 0.05 + 10.0 + 0.003)


def test_threshold_in_integer_microseconds():
    """T9-5: 8.3 ms + 50 us + 5647 us + 3 us = 14.000 ms exactly: not over 14 ms; 1 us more: over."""
    ok = trio(edge=row(age_r_ms=8.3), nxt=dict(row(), log_seq=11, timestamp_us=5_647))
    assert not T.ref_cond1(ok, 14.0)[0]
    over = trio(edge=row(age_r_ms=8.3), nxt=dict(row(), log_seq=11, timestamp_us=5_648))
    assert T.ref_cond1(over, 14.0)[1] == ['age_bound_over']


@pytest.mark.parametrize('where', ['edge', 'later'])
def test_missing_counter_in_the_window_is_bad(where):
    """A counter column missing (empty) on the edge row or a later row of the run is not 'unchanged'."""
    if where == 'edge':
        rows = trio(edge=row(psi_reset_count=''))
    else:
        rows = trio() + [dict(row(), log_seq=12, timestamp_us=20_000, reset_n='')]
    bad, why, _ = T.ref_cond1(rows, AGE_MAX)
    assert bad and any(w.endswith('_changed') for w in why)
