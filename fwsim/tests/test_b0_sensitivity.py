"""b0_sensitivity.judge: the criterion written before the run (STATUS_CODE L38). No DLL needed."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import b0_sensitivity as bs  # noqa: E402


def _row(v, k, done=True, e=0.05, le=True):
    return dict(v=v, b0_factor=k, completed=done, e_perp_max=e, e_perp_max_le_D=le)


def test_not_sensitive_when_close():
    rows = [_row(0.5, 1.0), _row(0.5, 0.7, e=0.07), _row(0.5, 1.3, e=0.03)]
    assert bs.judge(rows) == (False, [])


def test_sensitive_on_completion_change():
    s, why = bs.judge([_row(0.2, 1.0), _row(0.2, 0.7, done=False)])
    assert s and 'completed' in why[0]


def test_sensitive_on_d_status_change():
    s, why = bs.judge([_row(0.8, 1.0, e=0.09), _row(0.8, 1.3, e=0.11, le=False)])
    assert s and any('<= D' in w for w in why)


def test_sensitive_on_large_relative_change():
    s, why = bs.judge([_row(0.5, 1.0, e=0.04), _row(0.5, 0.85, e=0.061)])
    assert s and '%' in why[0]
