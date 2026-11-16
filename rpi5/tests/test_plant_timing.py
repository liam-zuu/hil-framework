"""plant_timing.py, batch 2 variants fault_plant_* (design 2026-11-15_Thiet_ke_bom_loi_dot2_v1.md 4.3): the timed
calls really run the plant fault path they are named after (the row of each variant, the contact points of the
zone ones counted call by call; Codex FQ1-3), zone_half keeps max(mu_hd) nominal (FQ1-4), and the script prints
them."""
import sys

import numpy as np
import pytest

import fault_plant
import plant_timing as pt
from params import MecanumParams

X0 = np.zeros(10)
X0[6:10] = 5.0
U = np.full(4, 0.3)
NAMES = ['fault_plant_base', 'fault_plant_idle', 'fault_plant_jam', 'fault_plant_sag', 'fault_plant_zone',
         'fault_plant_zone_half', 'fault_plant_all3']
FAULTS = {'fault_plant_idle': ['wheel_jam'], 'fault_plant_jam': ['wheel_jam'], 'fault_plant_sag': ['batt_sag'],
          'fault_plant_zone': ['mu_zone'], 'fault_plant_zone_half': ['mu_zone'],
          'fault_plant_all3': ['wheel_jam', 'batt_sag', 'mu_zone']}
ZONE = ('fault_plant_zone', 'fault_plant_zone_half', 'fault_plant_all3')


def run(n=40, monkeypatch=None):
    """Every variant called WARMUP + n times, as _time(fn, n) calls it; with monkeypatch, the calls of
    fault_plant.contact_points made by each variant are counted."""
    calls = {}
    cur = [None]
    if monkeypatch is not None:
        real = fault_plant.contact_points

        def counted(x, params):
            calls[cur[0]] = calls.get(cur[0], 0) + 1
            return real(x, params)
        monkeypatch.setattr(fault_plant, 'contact_points', counted)
    vs = pt.fault_plant_variants(n, X0, U)
    for name, fn, _ in vs:
        cur[0] = name
        for _ in range(pt.WARMUP + n):
            fn()
    return {name: (fn, plant) for name, fn, plant in vs}, calls


def rec(plant):
    d = plant.finish()
    assert d['overflow'] is False and d['rows'] <= len(plant._rec)
    return {c: plant._rec[:d['rows'], i] for i, c in enumerate(d['cols'])}


def test_names_and_the_row_of_each_variant():
    vs = pt.fault_plant_variants(10, X0, U)
    assert [v[0] for v in vs] == NAMES
    assert vs[0][2] is None
    for name, _, plant in vs[1:]:
        assert [r.fault for r in plant._p.schedule.rows] == FAULTS[name], name


def test_contact_points_every_call_of_the_zone_variants_only(monkeypatch):
    n = 40
    _, calls = run(n, monkeypatch)
    for name in NAMES:
        assert calls.get(name, 0) == (pt.WARMUP + n if name in ZONE else 0), name


def test_every_timed_call_is_inside_the_window_and_recorded():
    n = 40
    v, _ = run(n)
    nom = MecanumParams()
    for name in NAMES[2:]:
        r = rec(v[name][1])
        assert len(r['step']) == pt.WARMUP + n and np.all(r['step'] == pt.FAULT_PLANT_STEP), name
    j, s, z, h, a = (rec(v[f'fault_plant_{k}'][1]) for k in ('jam', 'sag', 'zone', 'zone_half', 'all3'))
    assert np.all(j['b_fl'] == nom.motor_b[0] + 0.1) and np.all(j['b_fr'] == nom.motor_b[1])
    assert np.all(j['mu_fl'] == nom.mu_hd[0]) and np.all(j['V_batt'] == nom.V_batt)
    w = n + pt.WARMUP + 100
    v_ramp = nom.V_batt + (7.44 - nom.V_batt) * (pt.FAULT_PLANT_STEP / w)
    assert np.all(s['V_batt'] == v_ramp) and 7.44 < v_ramp < nom.V_batt          # still on the ramp
    assert np.all(s['b_fl'] == nom.motor_b[0]) and np.all(s['mu_rr'] == nom.mu_hd[3])
    for c in ('mu_fl', 'mu_fr', 'mu_rl', 'mu_rr'):
        assert np.all(z[c] == 0.2) and np.all(a[c] == 0.2), c
    assert np.all(z['V_batt'] == nom.V_batt) and np.all(z['b_fl'] == nom.motor_b[0])
    assert np.all(a['b_fl'] == nom.motor_b[0] + 0.1) and np.all(a['V_batt'] == v_ramp)
    # zone_half: the left wheels (FL, RL) in the zone at every call, the right ones out: max(mu_hd) nominal
    assert np.all(h['mu_fl'] == 0.2) and np.all(h['mu_rl'] == 0.2)
    assert np.all(h['mu_fr'] == nom.mu_hd[1]) and np.all(h['mu_rr'] == nom.mu_hd[3])


def test_idle_is_the_nominal_path_and_faults_change_the_plant():
    v, _ = run()
    base = np.array(v['fault_plant_base'][0]())
    idle_plant = v['fault_plant_idle'][1]
    assert idle_plant._n == 0                                        # no row on: nothing recorded
    assert np.array_equal(np.array(v['fault_plant_idle'][0]()), base)  # same steps, bit for bit
    for name in NAMES[2:]:
        assert not np.array_equal(np.array(v[name][0]()), base), name


def test_window_and_record_sized_for_the_timed_calls():
    # built only, not run: the window and the record must hold WARMUP + n calls for any n (the run itself is
    # checked with n = 40 above), else the later timed calls would leave the fault path or overflow
    for n in (1, 20000):
        vs = pt.fault_plant_variants(n, X0, U)
        for name, _, plant in vs[2:]:
            assert len(plant._rec) >= pt.WARMUP + n, name
            assert all(r.end_ms > pt.FAULT_PLANT_STEP + pt.WARMUP + n - 1 for r in plant._p.schedule.rows)
            assert plant._act.plant_on, name


def test_main_prints_the_variants(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['plant_timing.py', '--no-rt', '--n', '20', '--nsub', '5'])
    assert pt.main() == 0
    out = capsys.readouterr().out
    lines = {ln.split()[0] for ln in out.splitlines() if ln.strip()}
    for name in NAMES:
        assert name in lines, name
    assert 'fault_all4' in lines and 'slipA_nsub5' in lines
