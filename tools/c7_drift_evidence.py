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
        for k in SESSIONS:
            if k == 's1_session_sha256' and no_clip:
                if k not in d or d[k] is not None:
                    raise ValueError(f'thieu_H1 requires {k} null')
            elif not isinstance(d.get(k), str) or not d[k]:
                raise ValueError(f'drift: missing {k}')

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
