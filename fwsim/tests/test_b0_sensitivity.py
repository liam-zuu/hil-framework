"""b0_sensitivity.judge: the criterion written before the run (STATUS_CODE L38), applied to known
results only (EV7-1, EV7-2: evaluate_v7 gives completed 'true' / 'false' / 'inconclusive' and a
status). No DLL needed."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import b0_sensitivity as bs  # noqa: E402


def _row(v, k, done='true', e=0.05, le=True, status='ok'):
    return dict(v=v, b0_factor=k, completed=done, status=status, e_perp_max=e, e_perp_max_le_D=le)


def test_not_sensitive_when_close():
    rows = [_row(0.5, 1.0), _row(0.5, 0.7, e=0.07), _row(0.5, 1.3, e=0.03)]
    assert bs.judge(rows) == (False, [], [])


def test_sensitive_on_completion_change():
    # strings, both truthy: a bool() comparison would miss 'true' -> 'false'
    s, why, und = bs.judge([_row(0.2, 1.0), _row(0.2, 0.7, done='false')])
    assert s and 'completed' in why[0] and und == []
    s, why, _ = bs.judge([_row(0.2, 1.0, done='false'), _row(0.2, 1.3)])
    assert s and 'false -> true' in why[0]


def test_sensitive_on_d_status_change():
    s, why, _ = bs.judge([_row(0.8, 1.0, e=0.09), _row(0.8, 1.3, e=0.11, le=False)])
    assert s and any('<= D' in w for w in why)


def test_sensitive_on_large_relative_change():
    s, why, _ = bs.judge([_row(0.5, 1.0, e=0.04), _row(0.5, 0.85, e=0.061)])
    assert s and '%' in why[0]


def test_inconclusive_or_track_error_is_undecided_not_judged():
    """EV7-1: false -> inconclusive is not evidence of sensitivity; two inconclusive runs are not
    evidence of no sensitivity; a completed run with a tracking error is not a known result."""
    s, why, und = bs.judge([_row(0.5, 1.0, done='false'), _row(0.5, 0.7, done='inconclusive', status='track_err')])
    assert (s, why, len(und)) == (False, [], 1) and 'not judged' in und[0]
    s, why, und = bs.judge([_row(0.5, 1.0, done='inconclusive', status='track_err', e=0.05),
                            _row(0.5, 0.7, done='inconclusive', status='track_err', e=0.20)])
    assert (s, why, len(und)) == (False, [], 1)                          # the 300 % change is not judged either
    s, why, und = bs.judge([_row(0.5, 1.0), _row(0.5, 1.3, status='track_err', e=0.20)])
    assert (s, why, len(und)) == (False, [], 1)


def test_factors_checked_before_the_run():
    """EV7-R-1: 1.0 required, no repeated factor, at least one factor other than 1.0."""
    assert bs.check_factors(list(bs.FACTORS)) is None
    assert bs.check_factors([1.0, 0.7]) is None
    assert 'include 1.0' in bs.check_factors([0.7, 1.3])
    assert 'repeat' in bs.check_factors([1.0, 1.0, 0.7])
    assert 'repeat' in bs.check_factors([1.0, 0.7, 0.7])
    assert 'other than 1.0' in bs.check_factors([1.0])


def test_n_pairs_counts_rows_with_a_factor_other_than_one():
    """EV7-R-1: the default grid, 3 speeds x 4 scaled factors = 12 pairs."""
    rows = [_row(v, k) for v in (0.2, 0.5, 0.8) for k in bs.FACTORS]
    assert bs.n_pairs(rows) == 12
    assert bs.n_pairs([_row(0.5, 1.0), _row(0.5, 0.7)]) == 1
    # rows minus speeds would count 2 here (Codex EV7-R-1: 1.0 repeated)
    assert bs.n_pairs([_row(0.5, 1.0), _row(0.5, 1.0), _row(0.5, 0.7)]) == 1


def test_known_pairs_still_judged_next_to_undecided_ones():
    rows = [_row(0.2, 1.0), _row(0.2, 0.7, done='inconclusive', status='track_err'), _row(0.2, 1.3, done='false')]
    s, why, und = bs.judge(rows)
    assert s and len(why) == 1 and 'b0x1.3' in why[0] and len(und) == 1 and 'b0x0.7' in und[0]
