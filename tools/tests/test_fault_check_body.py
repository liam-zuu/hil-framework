"""tools/fault_check.py check 11 with the s14 body loads body_drag, load_dist (design 2026-11-15 sections 3, 7.1,
14.10; Liam 17/11 CODE_FAULT2 "lam s14 ngay di"): a real run of the 1 kHz loop with body rows (alone, together,
with other plant rows) passes; each tampering of vx, vy, f_drag, f_dist in the plant record fails.
Run: python -m pytest tools/tests"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [os.path.join(ROOT, 'tools'), os.path.join(ROOT, 'rpi5'), HERE]
import fault_check as fc                       # noqa: E402
import fault_plant                             # noqa: E402
from test_fault_check_plant import C, _set, _tamper_record, open_loop   # noqa: E402

S14 = ('D1,body_drag,body,50,300,F=3.3\n'
       'L1,load_dist,body,150,400,F0=2.5;rate=0.8;Fmax=1.5;sigma=1.6;seed=3;dir_deg=30\n'
       'J1,wheel_jam,FL,100,200,b=0.1\nE1,enc_loss,RL,80,140,\n')


@pytest.fixture
def run(tmp_path):
    return open_loop(tmp_path, body=S14)


def test_cols_written_twice_agree():
    assert fc.REC_COLS == fault_plant.RECORD_COLS
    assert fc.DRAG_V == fault_plant.DRAG_V_MIN
    assert set(fc.PLANT_ROWS) == set(fault_plant.PLANT_FAULTS)


def test_good_body_run_passes(run):
    r = fc.check(*run)
    assert r['ok'], r['fails']
    pl = r['plant']
    assert pl['steps_ok'] and pl['values_bad'] == 0 and pl['pose_bad'] == 0 and pl['rows'] == 350   # [50, 400)
    import json
    man = json.loads(str(np.load(run[0])['manifest']))
    rec = np.load(os.path.join(os.path.dirname(run[0]), man['faults']['plant_record']['file']))['record']
    k = rec[:, 0].astype(int)
    assert np.all(rec[k < 150, C['f_dist']] == 0.0) and np.all(rec[k >= 150, C['f_dist']] != 0.0)
    assert np.all(rec[k >= 300, C['f_drag']] == 0.0) and np.any(rec[k < 300, C['f_drag']] == 3.3)


@pytest.mark.parametrize('body', [
    'D1,body_drag,body,0,400,F=3.3\n',
    'L1,load_dist,body,0,400,F0=1;rate=0;Fmax=0;sigma=0.5;seed=1;dir_deg=-45\n',
])
def test_one_body_row_passes(tmp_path, body):
    r = fc.check(*open_loop(tmp_path, body=body))
    assert r['ok'], r['fails']


@pytest.mark.parametrize('name,edit', [
    ('f_drag', _set('f_drag', 10, 1e-9)),
    ('f_drag off', _set('f_drag', 10, -3.3)),
    ('f_dist', _set('f_dist', 150, 1e-12)),
    ('f_dist before its window', _set('f_dist', 20, 0.5)),
    ('vx', _set('vx', 30, 0.01)),
    ('vy', _set('vy', 30, 0.01)),
])
def test_tampered_body_record_fails(run, tmp_path, name, edit):
    r = fc.check(*_tamper_record(run, tmp_path, edit))
    assert not r['ok'] and any('plant' in m for m in r['fails']), (name, r['fails'])


def test_drag_below_the_threshold_is_recorded_zero():
    """ref_body: F when hypot(vx, vy) >= 1e-3 at the start of the step, else 0."""
    import fault_schedule as fs
    rows = fs.parse(fs.MAGIC + '\nid,fault,target,start_ms,end_ms,params\nD1,body_drag,body,0,10,F=2\n')
    assert fc.ref_body(rows, [0], 3, (6e-4, 7e-4), {}) == (0.0, 0.0)
    assert fc.ref_body(rows, [0], 3, (6e-4, 8e-4), {}) == (2.0, 0.0)
    assert fc.ref_body(rows, [], 3, (1.0, 0.0), {}) == (0.0, 0.0)
