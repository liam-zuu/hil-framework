"""Clip H comparison fields shared by drift.json and the calib_drift block of gop."""
import math


V1 = 'c7_calib_drift_v1'
V2 = 'c7_calib_drift_v2'
BASE_FIELDS = ('D_H_mm', 'eps_H_mm', 'thieu_H1', 'vuot')
SESSIONS = ('s0_session_sha256', 's1_session_sha256')
V2_FIELDS = ('D_H_vi_tri_mm', 'D_H_huong_deg', 'eps_H_huong_deg', 'trang_thai_vi_tri',
             'trang_thai_huong', 'vuot_do_thieu')


def _number(value, name, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f'{name}: expected a finite number {">" if positive else ">="} 0')
    return value


def pose_triple(values, name):
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f'{name}: 3 numbers expected ({values!r})')
    try:
        values = [float(v) for v in values]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name}: expected three finite numbers') from exc
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f'{name}: expected three finite numbers')
    return values


def check_file(d, allowed_tasks):
    """Additional fields of the original so output (not included in the combined score)."""
    if not isinstance(d, dict) or d.get('kind') != 'c7_calib_drift':
        raise ValueError('not a c7_calib_drift file')
    for k in (*SESSIONS, 'tasks', 'k_start', 'tag_offset'):
        if k not in d:
            raise ValueError(f'drift: missing {k}')
    tasks = d['tasks']
    if not isinstance(tasks, list) or not tasks or any(t not in allowed_tasks for t in tasks):
        raise ValueError(f'drift: invalid tasks {tasks!r}')
    if d['k_start'] is not None or any(t.startswith('O1-K') for t in tasks):
        pose_triple(d['k_start'], 'drift k_start')
    pose_triple(d['tag_offset'], 'drift tag_offset')
    if 'version' not in d and any(k in d for k in V2_FIELDS):
        raise ValueError('drift: position and heading require the v2 version')
    return check(d)


def check(d):
    """Validate the recorded decision and return its schema version. Older gop blocks lack a version."""
    has_heading = any(k in d for k in V2_FIELDS)
    version = d.get('version', V2 if has_heading else V1)
    if version not in (V1, V2) or (version == V1 and has_heading):
        raise ValueError(f'inconsistent drift version {version!r}')
    fields = BASE_FIELDS + (V2_FIELDS if version == V2 else ())
    for k in fields:
        if k not in d:
            raise ValueError(f'drift: missing {k}')
    if type(d['thieu_H1']) is not bool or type(d['vuot']) is not bool:
        raise ValueError('thieu_H1 and vuot must be true/false')
    no_clip = d['thieu_H1']
    if 'version' in d or has_heading or any(k in d for k in SESSIONS):
        for key in SESSIONS:
            if key == 's1_session_sha256' and no_clip:
                if key not in d or d[key] is not None:
                    raise ValueError(f'thieu_H1 requires {key} null')
            elif not isinstance(d.get(key), str) or not d[key]:
                raise ValueError(f'drift: missing {key}')

    limit = _number(d['eps_H_mm'], 'eps_H_mm', positive=True)
    position = d['D_H_mm']
    if version == V2:
        heading = d['D_H_huong_deg']
        missing = position is None or heading is None
        if d['vuot_do_thieu'] is not missing:
            raise ValueError('vuot_do_thieu must indicate a missing quantity')
        for state, value in (('trang_thai_vi_tri', position), ('trang_thai_huong', heading)):
            if d[state] != ('thieu' if value is None else 'so'):
                raise ValueError(f'{state} disagrees with the measured value')
        if isinstance(d['D_H_vi_tri_mm'], bool) or d['D_H_vi_tri_mm'] != position:
            raise ValueError('D_H_vi_tri_mm differs from D_H_mm')
        if no_clip and heading is not None:
            raise ValueError('thieu_H1 requires D_H_huong_deg null')
        if heading is not None:
            _number(heading, 'D_H_huong_deg')
        heading_limit = d['eps_H_huong_deg']  # null until the heading threshold is locked
        if heading_limit is not None:
            _number(heading_limit, 'eps_H_huong_deg', positive=True)
        over = missing or (heading_limit is not None and heading > heading_limit)
    else:
        if position is None and not no_clip:
            raise ValueError('v1: D_H_mm required when H1 is present')
        over = no_clip
    if no_clip and position is not None:
        raise ValueError('thieu_H1 requires D_H_mm null')
    if position is not None:
        over = (_number(position, 'D_H_mm') > limit) or over
    if d['vuot'] != over:
        raise ValueError(f'vuot {d["vuot"]} differs from {over} by the measured drift and thresholds')
    return version
