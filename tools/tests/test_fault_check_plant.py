"""tools/fault_check.py check 11 (batch 2 plant faults, design 2026-11-15 section 7.1): a real run of the 1 kHz
loop (simulate_hw) with plant rows passes; each tampering of the plant record fails: a value, a missing or extra
step, the mask, the nominal values, the pose, the SHA, the file; a plant record without plant rows fails; a
finite run cut at the stop step fails for that reason only (batch 1 rule), with its record right up to the stop
step (Codex FP2-2).
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


def open_loop(tmp_path, body=BODY, steps=400, name='frames', argv=('run_hil.py',)):
    csv_ = tmp_path / f'{name}.csv'
    csv_.write_text(HEAD + body)
    sched = fs.load(str(csv_), steps, 'plant', 'traction')
    p = str(tmp_path / f'{name}.npz')
    node = HILNode(moving_plant(), config=HILConfig(simulate_hw=True, max_steps=steps, frame_log=p, faults=sched))
    node._spi.transfer = good_miso
    node.run()
    return p, str(csv_), with_argv(write_step_log(str(tmp_path / f'{name}_steps.csv'), steps), argv)


def with_argv(step_log, argv=('run_hil.py',)):
    """run_hil.py keeps its argv in the step log .json (check 11 reads --mu from it, Codex FP1-2 round 2)."""
    with open(step_log + '.json') as fh:
        meta = json.load(fh)
    meta['argv'] = list(argv)
    with open(step_log + '.json', 'w') as fh:
        json.dump(meta, fh)
    return step_log


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
    from plant_blob import plant_source
    src = plant_source()
    assert fc.nominal_problems(nom, src, ['run_hil.py', '--mu', '0.3']) == []
    assert fc.nominal_problems(nom, src, ['run_hil.py', '--mu=0.3', '--mock']) == []
    assert fc.nominal_problems(nom, None, ['run_hil.py', '--mu', '0.3'])     # no plant_src: cannot be checked
    assert fc.nominal_problems(nom, src, ['run_hil.py'])                     # four equal values without --mu
    assert fc.nominal_problems(nom, src, None)                               # argv unknown
    assert fc.nominal_problems(nom, src, ['run_hil.py', '--mu', '0.2'])
    nom['mu_hd'] = [0.3, 0.3, 0.2, 0.3]
    assert fc.nominal_problems(nom, src, ['run_hil.py', '--mu', '0.3'])


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
    with_argv(steps)
    man = json.loads(str(np.load(p)['manifest']))
    assert man['faults']['cut'] is True                             # RUN ended before the window
    r = fc.check(p, csv_, steps)
    # FP1-12: the run fails because its schedule was cut (rule of batch 1), and only for that; the plant record
    # itself is right: steps up to the stop step, values, pose
    assert not r['ok']
    assert r['fails'] == ['the schedule did not run in full (a window cut or never started)'], r['fails']   # FP2-2
    assert r['plant']['steps_ok'] and r['plant']['values_bad'] == 0 and r['plant']['pose_bad'] == 0
    t0, stop = r['t0_step'], r['stop_step']
    rec = np.load(_rec_path(p))['record']
    assert list(rec[:, 0].astype(int)) == list(range(t0 + 20, stop))


def test_c7_table_pairs_a_plant_fault_run_with_its_baseline():
    import fault_c7_table as t
    run = dict(plant='FaultPlant', faults=dict(plant_record=dict(file='x', inner='MecanumPlant')))
    assert t._man_value(run, 'plant') == t._man_value(dict(plant='MecanumPlant'), 'plant') == 'MecanumPlant'
    assert t._man_value(dict(plant='FaultPlant', faults=dict()), 'plant') == 'FaultPlant'   # no record: not paired
    assert t._man_value(dict(plant='FaultPlant', faults=None), 'plant') == 'FaultPlant'
    other = dict(plant='FaultPlant', faults=dict(plant_record=dict(file='x', inner='OtherPlant')))
    assert t._man_value(other, 'plant') == 'OtherPlant'               # FP1-11: the recorded class, not a guess
    assert t._man_value(dict(plant='FaultPlant', faults=dict(plant_record=dict(file='x'))), 'plant') == 'FaultPlant'
    assert t._man_value(run, 'imu_source') is None


# ── Codex vòng 1 (FP1-1, 2, 3, 4, 9, 11) ────────────────────────────────────
@pytest.mark.parametrize('zone', ['x0=0.12;x1=1;y0=-1;y1=1', 'x0=-0.12;x1=0.12;y0=-1;y1=1',
                                  'x0=-1;x1=1;y0=0.12;y1=1', 'x0=-1;x1=1;y0=-0.12;y1=0.12'])
def test_zone_on_the_hub_bounds_passes(tmp_path, zone):              # FP1-1: hubs exactly on a bound at step 0
    p, csv_, steps = open_loop(tmp_path, body=f'Z1,mu_zone,all,0,1,{zone};mu=0.05\n', steps=50, name='z')
    r = fc.check(p, csv_, steps)
    assert r['ok'], r['fails']
    rec = np.load(_rec_path(p))['record']
    assert sum(rec[0, C['mu_fl']:C['mu_rr'] + 1] == 0.05) == 2       # two hubs on the closed side, two out


def test_mask_and_rows_strict(run, tmp_path):                        # FP1-3
    r = fc.check(*_tamper_record(run, tmp_path, _set('mask', 0, 0.5)))
    assert not r['ok']
    src, csv_, steps = run
    bad = resave(src, str(tmp_path / 'rows.npz'),
                 lambda d, m: m['faults']['plant_record'].update(rows=float(m['faults']['plant_record']['rows'])))
    assert not fc.check(bad, csv_, steps)['ok']


def test_large_values_compare_by_ulp():                              # FP1-4
    from fractions import Fraction
    from params import MecanumParams
    import fault_plant as fp
    base = MecanumParams()
    r = fs.parse(HEAD + 'J1,wheel_jam,FL,0,1,tc=10000.006172835;b=12345.678901234\n')
    p, _ = fp.PlantVariant(r, base).at((0,), 0, [0, 0, 0])
    nom = dict(V_batt=base.V_batt, lx=base.lx, ly=base.ly, mu_hd=[float(v) for v in base.mu_hd],
               **{k: [float(v) for v in getattr(base, k)] for k in ('motor_b', 'motor_Tc', 'motor_Ts')})
    V, mu, b, tc, ts = fc.ref_plant(r, [0], 0, (0.0, 0.0, 0.0), nom)
    assert fc._close(p.motor_Tc[0], tc[0]) and fc._close(p.motor_Ts[0], ts[0]) and fc._close(p.motor_b[0], b[0])
    assert not fc._close(p.motor_Tc[0] + 1e-6, tc[0]) and not fc._close(float('nan'), Fraction(1))


def test_nominal_tied_to_the_run_source(run, tmp_path):              # FP1-2
    src, csv_, steps = run
    for name, edit in (('src', lambda d, m: m['plant_src'].update(params_py='0' * 40)),
                       ('no src', lambda d, m: m.pop('plant_src')),
                       ('nan', lambda d, m: m['faults']['plant_record']['nominal'].update(mu_hd=[float('nan')] * 4)),
                       ('zero', lambda d, m: m['faults']['plant_record']['nominal'].update(mu_hd=[0.0] * 4))):
        r = fc.check(resave(src, str(tmp_path / f'{name}.npz'), edit), csv_, steps)
        assert not r['ok'] and any('nominal' in m or 'params.py' in m for m in r['fails']), (name, r['fails'])


def test_record_file_must_be_a_bare_name(run, tmp_path):             # FP1-9
    src, csv_, steps = run
    rp = _rec_path(src)
    sub = tmp_path / 'other'
    sub.mkdir()
    import shutil
    shutil.copy2(rp, sub / 'rec.npz')
    for name, f in (('dotdot', '../' + os.path.basename(rp)), ('abs', str(sub / 'rec.npz')),
                    ('sub', 'other/rec.npz')):
        bad = resave(src, str(tmp_path / f'{name}.npz'), lambda d, m: m['faults']['plant_record'].update(file=f))
        assert not fc.check(bad, csv_, steps)['ok'], name


def test_record_inner_must_be_mecanum(run, tmp_path):                # FP1-11
    src, csv_, steps = run
    bad = resave(src, str(tmp_path / 'i.npz'), lambda d, m: m['faults']['plant_record'].update(inner='OtherPlant'))
    assert not fc.check(bad, csv_, steps)['ok']


# ── Codex vòng 2 (FP1-2 còn, FP2-1) ─────────────────────────────────────────
def test_low_battery_ramp_passes(tmp_path):                          # FP2-1: v1 = 0.1, cancellation near v1
    p, csv_, steps = open_loop(tmp_path, body='B1,batt_sag,batt,0,1001,v1=0.1;ramp_ms=1000\n', steps=1100,
                               name='batt')
    r = fc.check(p, csv_, steps)
    assert r['ok'], r['fails']
    rec = np.load(_rec_path(p))['record']
    assert rec[990, C['V_batt']] == 12.4 + (0.1 - 12.4) * (990 / 1000)


def test_nominal_mu_and_record_mu_forged_together_fail(run, tmp_path):   # FP1-2 round 2
    def edit(rec):
        m = rec[:, C['mu_fl']:C['mu_rr'] + 1]
        rec[:, C['mu_fl']:C['mu_rr'] + 1] = np.where(m == 0.436, 0.3, m)
        return rec
    src, csv_, steps = _tamper_record(run, tmp_path, edit)
    bad = resave(src, str(tmp_path / 'forged.npz'),
                 lambda d, m: m['faults']['plant_record']['nominal'].update(mu_hd=[0.3] * 4))
    r = fc.check(bad, csv_, steps)
    assert not r['ok'] and any('mu_hd' in m for m in r['fails']), r['fails']


def test_run_mu_reads_the_parser():
    assert fc.run_mu(['run_hil.py']) is None and fc.run_mu(['x', '--mu', '0.25']) == 0.25
    with pytest.raises(ValueError):
        fc.run_mu(['x', '--mu', 'abc'])
    with pytest.raises(ValueError):
        fc.run_mu(None)


def test_argv_of_the_step_log_decides_the_nominal_mu(tmp_path):      # FP1-2 round 2: --mu read from the run
    body = 'J1,wheel_jam,FL,10,50,b=0.1\n'
    p, csv_, steps = open_loop(tmp_path, body=body, steps=100, name='mu_ok', argv=('run_hil.py', '--mu', '0.436'))
    assert fc.check(p, csv_, steps)['ok']
    p, csv_, steps = open_loop(tmp_path, body=body, steps=100, name='mu_bad', argv=('run_hil.py', '--mu', '0.3'))
    r = fc.check(p, csv_, steps)                                     # the plant ran 0.436, its argv says 0.3
    assert not r['ok'] and any('mu_hd' in m for m in r['fails']), r['fails']
