"""
run_validity.py: checks that decide whether one run can be used at all, shared by the
analysis scripts (tools/s02_metrics.py for S02 SIM/HIL/REAL logs, fwsim/o1_feasibility.py,
and the S03/S04 evaluation on fwsim).

pos_reset_mid (STATUS_CODE L37, Liam 05/11 CODE session: spec controller-IMU v2.6 section
14.7.3 = (A)): the firmware sets the pose back to 0 every time run_active rises
(pos_ctrl.c pos_step, pos_reset_n += 1). run_active is motor on + MODE_AUTO + a fresh
trajectory packet (main.c), so a run whose packets go stale and come back has its pose reset
in the middle; that run is broken and is run again. The protocol is not changed (A1).

One log or one simulation is taken as one run (the same assumption as the metric window of
s02_metrics.py): the run spans the first to the last line with run_active = 1, and every
increase of pos_reset_n inside that span is a mid-run reset. The reset at the start of the
run happens before its first active line is logged (pos_step runs before the logger), so it
is not counted.
"""
import numpy as np


def pos_reset_mid(run_active, reset_n):
    """Number of pose resets inside the run, or None when the log cannot show it.

    run_active, reset_n: per-line arrays (NaN = empty field). None when either column is
    missing or empty (firmware before pos_ctrl.c / run_active, 05/11), or when no line has
    run_active = 1. Lines with an empty pos_reset_n inside the span are skipped; the counter
    only grows, so a lost line cannot hide a reset that later lines still show.
    """
    if run_active is None or reset_n is None:
        return None
    ra = np.asarray(run_active, dtype=float)
    rn = np.asarray(reset_n, dtype=float)
    if ra.shape != rn.shape or not np.isfinite(ra).any() or not np.isfinite(rn).any():
        return None
    act = np.flatnonzero(ra == 1)
    if act.size == 0:
        return None
    span = rn[act[0]:act[-1] + 1]
    span = span[np.isfinite(span)]
    if span.size == 0:
        return None
    if np.any(np.diff(span) < 0):
        raise ValueError('pos_reset_n goes back inside the run: a DUT restart or a bad line')
    return int(span[-1] - span[0])


def run_bad_fields(k):
    """Columns for a metrics / summary row from pos_reset_mid(): pos_reset_mid (count, NaN
    when not checkable), run_bad (1 broken, 0 usable, NaN not checkable), run_bad_reason."""
    if k is None:
        return dict(pos_reset_mid=np.nan, run_bad=np.nan, run_bad_reason='pos_reset_n not logged')
    return dict(pos_reset_mid=k, run_bad=int(k > 0),
                run_bad_reason=f'pos_reset_n +{k} mid-run (L37): run again' if k > 0 else '')
