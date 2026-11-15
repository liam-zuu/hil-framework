"""tools/fault_check.py check 11 (batch 2 plant faults, design 2026-11-15 section 7.1): a real run of the 1 kHz
loop (simulate_hw) with plant rows passes; each tampering of the plant record fails: a value, a missing or extra
step, the mask, the nominal values, the pose, the SHA, the file; a plant record without plant rows fails; a
finite run cut at the stop step passes with the record cut there too.
Run: python -m pytest tools/tests"""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [os.path.join(ROOT, 'tools'), os.path.join(ROOT, 'rpi5'), HERE]
import fault_check as fc                       # noqa: E402
import fault_schedule as fs                    # noqa: E402
from hil_node import HILConfig, HILNode        # noqa: E402
from test_fault_check import HEAD, good_miso, moving_plant, resave, write_step_log   # noqa: E402

BODY = ('J1,wheel_jam,FL+RR,50,250,b=0.1;tc=0.02\nB1,batt_sag,batt,100,350,v1=8;ramp_ms=150\n'
        'M1,mu_time,FR,120,200,mu=0.1\nZ1,mu_zone,all,200,400,x0=-0.5;x1=0.0;y0=-1;y1=1;mu=0.05\n'
        'E1,enc_loss,RL,80,140,\nS1,gyro_scale,gz,150,300,s=0.3\n')


def open_loop(tmp_path, body=BODY, steps=400, name='frames'):
    csv_ = tmp_path / f'{name}.csv'
    csv_.write_text(HEAD + body)
    sched = fs.load(str(csv_), steps, 'plant', 'traction')
    p = str(tmp_path / f'{name}.npz')
    node = HILNode(moving_plant(), config=HILConfig(simulate_hw=True, max_steps=steps, frame_log=p, faults=sched))
    node._spi.transfer = good_miso
    node.run()
    return p, str(csv_), write_step_log(str(tmp_path / f'{name}_steps.csv'), steps)


@pytest.fixture
def run(tmp_path):
    return open_loop(tmp_path)


def test_good_plant_run_passes(run):
    r = fc.check(*run)
    assert r['ok'], r['fails']
    pl = r['plant']
    assert pl['steps_ok'] and pl['values_bad'] == 0 and pl['pose_bad'] == 0 and pl['rows'] == 350   # [50, 400)
    assert r['frames_injected'] > 0                                    # the frame rows still patch


def _rec_path(npz):
    man = json.loads(str(np.load(npz)['manifest']))
    return os.path.join(os.path.dirname(npz), man['faults']['plant_record']['file'])


def _tamper_record(run, tmp_path, edit, fix_sha=True):
    """Copy the run, change the plant record with edit(rec), write it back and (fix_sha) its SHA in the
    manifest, so that only the edited content can fail."""
    src, csv_, steps = run
    dst = str(tmp_path / 't.npz')
    g = np.load(_rec_path(src))
    rec = edit(g['record'].copy())
    rp = str(tmp_path / 't_plant_faults.npz')
    np.savez(rp, record=rec, cols=g['cols'])
    sha = hashlib.sha256(open(rp, 'rb').read()).hexdigest()

    def man_edit(d, m):
        m['faults']['plant_record'].update(file=os.path.basename(rp), rows=len(rec))
        if fix_sha:
            m['faults']['plant_record']['sha256'] = sha
    resave(src, dst, man_edit)
    return dst, csv_, steps


C = {n: i for i, n in enumerate(fc.REC_COLS)}


def _set(col, k_idx, delta):
    def edit(rec):
        rec[k_idx, C[col]] += delta
        return rec
    return edit


@pytest.mark.parametrize('name,edit', [
    ('b value', _set('b_fl', 10, 1e-6)),
    ('tc value', _set('tc_rr', 10, 1e-6)),
    ('ts value', _set('ts_fl', 10, 1e-6)),
    ('V ramp', _set('V_batt', 120, 1e-6)),
    ('mu', _set('mu_fr', 100, -0.01)),
    ('mask', _set('mask', 5, 2)),
    ('pose x', _set('x', 30, 0.01)),
    ('pose theta', _set('theta', 30, 0.01)),
    ('step missing', lambda rec: np.delete(rec, 40, axis=0)),
    ('step extra', lambda rec: np.vstack([rec, rec[-1:] + [[1] + [0] * (rec.shape[1] - 1)]])),
    ('step shifted', _set('step', 0, -1)),
])
def test_tampered_record_fails(run, tmp_path, name, edit):
    r = fc.check(*_tamper_record(run, tmp_path, edit))
    assert not r['ok'] and any('plant' in m for m in r['fails']), (name, r['fails'])


def test_unchanged_copy_passes(run, tmp_path):
    r = fc.check(*_tamper_record(run, tmp_path, lambda rec: rec))
    assert r['ok'], r['fails']


def test_sha_file_and_nominal(run, tmp_path):
    # a record changed after the run, its manifest SHA left as it was
    r = fc.check(*_tamper_record(run, tmp_path, _set('b_fl', 10, 1e-6), fix_sha=False))
    assert not r['ok'] and any('sha256' in m for m in r['fails'])
    src, csv_, steps = run
    for name, edit in (('sha', lambda d, m: m['faults']['plant_record'].update(sha256='0' * 64)),
                       ('missing', lambda d, m: m['faults']['plant_record'].update(file='nope.npz')),
                       ('no record', lambda d, m: m['faults'].pop('plant_record')),
                       ('nominal V', lambda d, m: m['faults']['plant_record']['nominal'].update(V_batt=12.0)),
                       ('nominal b', lambda d, m: m['faults']['plant_record']['nominal']['motor_b'].__setitem__(0, 1.0)),
                       ('overflow', lambda d, m: m['faults']['plant_record'].update(overflow=True))):
        r = fc.check(resave(src, str(tmp_path / f'{name}.npz'), edit), csv_, steps)
        assert not r['ok'], name


def test_nominal_mu_from_dash_mu_is_accepted():
    from params import MecanumParams
    p = MecanumParams()
    nom = dict(V_batt=p.V_batt, lx=p.lx, ly=p.ly, mu_hd=[0.3] * 4,
               **{k: [float(v) for v in getattr(p, k)] for k in ('motor_b', 'motor_Tc', 'motor_Ts')})
    assert fc.nominal_problems(nom) == []
    nom['mu_hd'] = [0.3, 0.3, 0.2, 0.3]
    assert fc.nominal_problems(nom)


def test_record_without_plant_rows_fails(tmp_path):
    p, csv_, steps = open_loop(tmp_path, body='E1,enc_loss,RL,80,140,\n', name='sig')
    r = fc.check(p, csv_, steps)
    assert r['ok'] and r['plant'] is None
    bad = resave(p, str(tmp_path / 'b.npz'), lambda d, m: m['faults'].update(plant_record=dict(file='x')))
    assert not fc.check(bad, csv_, steps)['ok']


def test_finite_run_record_cut_at_stop(tmp_path):
    from test_fault_check import finite_run
    import fault_schedule
    real_load = fault_schedule.load
    # finite_run loads with the static IMU and no plant source: plant rows need 'traction'
    fault_schedule.load = lambda path, run_ms, imu, plant=None: real_load(path, run_ms, imu, 'traction')
    try:
        p, csv_, steps, node = finite_run(tmp_path, 'J1,wheel_jam,FL,20,140,b=0.1\n', after=0.06)
    finally:
        fault_schedule.load = real_load
    man = json.loads(str(np.load(p)['manifest']))
    assert man['faults']['cut'] is True                             # RUN ended before the window
    r = fc.check(p, csv_, steps)
    assert r['plant']['steps_ok'] and r['plant']['values_bad'] == 0
    t0, stop = r['t0_step'], r['stop_step']
    rec = np.load(_rec_path(p))['record']
    assert int(rec[0, 0]) == t0 + 20 and int(rec[-1, 0]) == stop - 1


def test_c7_table_pairs_a_plant_fault_run_with_its_baseline():
    import fault_c7_table as t
    run = dict(plant='FaultPlant', faults=dict(plant_record=dict(file='x')))
    assert t._man_value(run, 'plant') == t._man_value(dict(plant='MecanumPlant'), 'plant') == 'MecanumPlant'
    assert t._man_value(dict(plant='FaultPlant', faults=dict()), 'plant') == 'FaultPlant'   # no record: not paired
    assert t._man_value(dict(plant='FaultPlant', faults=None), 'plant') == 'FaultPlant'
    assert t._man_value(run, 'imu_source') is None
