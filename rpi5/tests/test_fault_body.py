"""Fault injection batch 2, s14 body loads body_drag and load_dist (design 03_CODE/docs/
2026-11-15_Thiet_ke_bom_loi_dot2_v1.md of the thesis repo, sections 3, 14.10; Liam 17/11 CODE_FAULT2 "lam s14
ngay di"): schedule rows, the force, the plant step with f_ext, the Pi wrapper, the record, hil_node, run_hil."""
import json
import math

import numpy as np
import pytest

import fault_plant as fp
import fault_schedule as fs
from hil_node import HILConfig, HILNode
from mecanum_plant import MecanumPlant
from params import MecanumParams
from plant_step import plant_step
from test_fault_plant import DUTY, HEAD, MOVING, _args, _Run, plain, rows, sched

DIST = 'L1,load_dist,body,{a},{b},F0=2.5;rate=0.8;Fmax=1.5;sigma=1.6;seed=7;dir_deg=90\n'


# ── schedule ────────────────────────────────────────────────────────────────
def test_parse_body_rows():
    r = rows('D1,body_drag,body,10,900,F=3.3\n' + DIST.format(a=0, b=1000))
    assert [x.fault for x in r] == ['body_drag', 'load_dist']
    assert r[0].p('F') == fs.Fraction('3.3') and r[1].p('seed') == 7 and r[1].p('dir_deg') == 90
    assert r[0].wheels == () and fs.has_plant_rows(r) and fs.BATCH2 == ()
    assert set(fs.BODY_FAULTS) <= set(fs.PLANT_FAULTS) and not set(fs.BODY_FAULTS) & set(fs.FRAME_FAULTS)


@pytest.mark.parametrize('body', [
    'D1,body_drag,body,1,9,F=0\n',                                     # F > 0
    'D1,body_drag,body,1,9,F=-1\n',
    'D1,body_drag,FL,1,9,F=1\n',                                       # target body only
    'D1,body_drag,all,1,9,F=1\n',
    'L1,load_dist,body,1,9,F0=0;rate=0;Fmax=0;sigma=0;seed=1;dir_deg=0\n',      # not all 0
    'L1,load_dist,body,1,9,F0=-1;rate=0;Fmax=0;sigma=0;seed=1;dir_deg=0\n',     # >= 0
    'L1,load_dist,body,1,9,F0=1;rate=0;Fmax=0;sigma=-1;seed=1;dir_deg=0\n',
    'L1,load_dist,body,1,9,F0=1;rate=1;Fmax=0;sigma=0;seed=1;dir_deg=0\n',      # a ramp needs Fmax > 0
    'L1,load_dist,body,1,9,F0=1;rate=0;Fmax=1;sigma=0;seed=1;dir_deg=0\n',      # Fmax without a ramp
    'D1,body_drag,body,1,9,F=1\nD2,body_drag,body,5,20,F=2\n',         # two drags at once
    'L1,load_dist,body,1,9,F0=1;rate=0;Fmax=0;sigma=0;seed=1;dir_deg=0\n'
    'L2,load_dist,body,8,20,F0=1;rate=0;Fmax=0;sigma=0;seed=2;dir_deg=0\n',     # two loads at once
])
def test_check_refuses(body):
    with pytest.raises(fs.ScheduleError):
        rows(body)


@pytest.mark.parametrize('plant', [None])
def test_body_rows_need_the_traction_plant(plant):
    with pytest.raises(fs.ScheduleError):
        rows('D1,body_drag,body,1,9,F=1\n', plant=plant)


def test_drag_and_load_overlap_and_join_other_plant_rows():
    r = rows('D1,body_drag,body,0,500,F=3.3\n' + DIST.format(a=100, b=600) + 'B1,batt_sag,batt,200,700,v1=8;ramp_ms=0\n'
             'D2,body_drag,body,500,800,F=1\n')                       # back to back: [0, 500) then [500, 800)
    assert len(r) == 4


def test_check_params_needs_finite_body_values():
    p = MecanumParams()
    for body in ('D1,body_drag,body,1,9,F=1e99\n', 'D1,body_drag,body,1,9,F=1\n'):
        fp.check_params(rows(body), p)
    big = '9' * 400
    r = fs.parse(HEAD + f'D1,body_drag,body,1,9,F={big}\n')
    with pytest.raises(fs.ScheduleError):
        fp.check_params(r, p)


# ── the force ───────────────────────────────────────────────────────────────
def test_body_step_without_force_is_plant_step_bit_for_bit():
    p = MecanumParams()
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = rng.normal(size=10) * np.array([2, 2, 3, 0.5, 0.5, 1, 8, 8, 8, 8])
        u = rng.uniform(-1, 1, size=4)
        assert np.array_equal(fp.body_step(x, u, p, 0.001, None), plant_step(x, u, p, 0.001, None))
        assert np.array_equal(fp.body_step(list(x), list(u), p, 0.001, None), plant_step(x, u, p, 0.001, None))


def test_drag_force_against_the_velocity_and_off_below_the_threshold():
    r = rows('D1,body_drag,body,0,100,F=3.3\n')
    bf = fp.BodyForce(r)
    x = np.zeros(10)
    x[3], x[4] = 0.3, -0.4
    f, fd, fdist = bf.at((0,), 5, x)
    assert fd == 3.3 and fdist == 0.0
    gx, gy = f(0.0, 0.0, 0.3, -0.4)
    assert math.isclose(gx, -3.3 * 0.6) and math.isclose(gy, 3.3 * 0.8)
    assert f(0.0, 0.0, 0.0, 0.0) == (0.0, 0.0) and f(1.0, 1.0, 9e-4, 0.0) == (0.0, 0.0)
    assert f(0.0, 0.0, 1e-3, 0.0)[0] == -3.3                      # at the threshold the drag acts
    x[3], x[4] = 5e-4, 5e-4                                         # |v| < 1e-3 at the start of the step
    assert bf.at((0,), 5, x)[1] == 0.0
    assert bf.at((), 5, x) == (None, 0.0, 0.0)


def test_load_value_ramp_cap_noise_and_direction():
    r = rows(DIST.format(a=100, b=5100), run_ms=6000)[0]
    n = fp.dist_noise(r)
    assert len(n) == 5000 and np.array_equal(n, np.random.default_rng(7).standard_normal(5000))
    for k in (100, 101, 1000, 1974, 1975, 2000, 5099):
        d = k - 100
        want = (2.5 + min(1.5, (0.8 * d) / 1000.0)) + 1.6 * n[d]
        assert fp.dist_value(r, k, n) == want, k
    assert fp.dist_value(r, 100, n) - 1.6 * n[0] == 2.5            # no ramp at the start
    assert math.isclose(fp.dist_value(r, 5000, n) - 1.6 * n[4900], 4.0)   # capped at F0 + Fmax
    c, s = fp.dist_dir(r)
    assert abs(c) < 1e-15 and s == 1.0
    bf = fp.BodyForce([r])
    f, fd, fdist = bf.at((0,), 2000, np.zeros(10))
    assert fd == 0.0 and fdist == fp.dist_value(r, 2000, n)
    assert f(0.0, 0.0, 1.0, 1.0) == (fdist * c, fdist * s)          # no drag: the velocity does not matter


def test_load_noise_does_not_depend_on_the_steps_run_nor_touch_the_plant_rng():
    r = rows(DIST.format(a=0, b=1000))
    a, b = fp.BodyForce(r), fp.BodyForce(r)
    va = [a.at((0,), k, np.zeros(10))[2] for k in range(1000)]
    vb = [b.at((0,), k, np.zeros(10))[2] for k in (999, 3, 500)]       # other steps, other order
    assert vb == [va[999], va[3], va[500]]
    other = fp.BodyForce(rows(DIST.format(a=0, b=1000).replace('seed=7', 'seed=8')))
    assert [other.at((0,), k, np.zeros(10))[2] for k in range(5)] != va[:5]
    run = _Run(DIST.format(a=0, b=1000))
    before = run.inner._rng.bit_generator.state
    run.go(200)
    assert run.inner._rng.bit_generator.state == before


def test_drag_and_load_together_add_up():
    r = rows('D1,body_drag,body,0,100,F=2\n' + DIST.format(a=0, b=100).replace('sigma=1.6', 'sigma=0'))
    f, fd, fdist = fp.BodyForce(r).at((0, 1), 0, np.r_[0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0])
    assert fd == 2.0 and fdist == 2.5
    gx, gy = f(0.0, 0.0, 1.0, 0.0)
    assert gx == pytest.approx(-2.0) and gy == pytest.approx(2.5)


# ── the Pi wrapper ──────────────────────────────────────────────────────────
def test_no_body_row_on_is_bit_identical_to_the_plain_plant():
    r = _Run('D1,body_drag,body,700,900,F=3.3\n' + DIST.format(a=800, b=1000))
    assert np.array_equal(r.go(700), plain(700))
    assert r.plant._n == 0


def test_body_step_is_the_traction_step_with_the_force():
    """FaultPlant with a body row: x_{k+1} = _traction_step(x_k, u, p, dt, f_ext=f_k), f_k the BodyForce of step
    k, the tick counter as MecanumPlant.step."""
    body = 'D1,body_drag,body,0,300,F=3.3\n' + DIST.format(a=100, b=300)
    r = _Run(body)
    got = r.go(300)
    bf = fp.BodyForce(rows(body))
    p = MecanumParams()
    x = np.array(MOVING, dtype=float)
    for k in range(300):
        on = tuple(i for i, w in enumerate(((0, 300), (100, 300))) if w[0] <= k < w[1])
        f = bf.at(on, k, x)[0]
        x = fp.body_step(x, [0.3] * 4, p, 0.001, f)
        assert np.array_equal(got[k], x), k
    assert r.inner.ticks == 300


def test_drag_stops_the_car_sooner():
    """Smoke physics: rolling with the motors off (duty 0), the drag takes the body to rest sooner and nearer."""
    n = 1500
    s0 = plain(n, duty=(0, 0, 0, 0))
    s = _Run('D1,body_drag,body,0,1500,F=3.3\n', run_ms=1500).go(n, duty=(0, 0, 0, 0))
    assert s[-1, 0] < s0[-1, 0]
    v, v0 = np.hypot(s[:, 3], s[:, 4]), np.hypot(s0[:, 3], s0[:, 4])
    assert np.argmax(v < 1e-3) < np.argmax(v0 < 1e-3) or v0[-1] >= 1e-3
    assert np.all(v[100:] <= v0[100:] + 1e-12)


def test_load_pushes_the_body_along_its_direction():
    s0 = plain(800)
    s = _Run(DIST.format(a=0, b=1000).replace('sigma=1.6', 'sigma=0')).go(800)
    assert s[-1, 1] > s0[-1, 1] + 1e-3                                  # dir 90: pushed to +y


def test_record_of_body_rows(tmp_path):
    path = str(tmp_path / 'r_plant_faults.npz')
    body = 'D1,body_drag,body,10,60,F=3.3\n' + DIST.format(a=40, b=90) + 'B1,batt_sag,batt,50,70,v1=10;ramp_ms=0\n'
    r = _Run(body, path=path, run_ms=200)
    states = r.go(120)
    pr = r.patcher.summary(last_step=119)['plant_record']
    rec = np.load(path)['record']
    c = {k: i for i, k in enumerate(fp.RECORD_COLS)}
    assert fp.RECORD_COLS[-4:] == ('vx', 'vy', 'f_drag', 'f_dist') and pr['cols'] == list(fp.RECORD_COLS)
    steps = rec[:, 0].astype(int)
    assert list(steps) == list(range(10, 90)) and pr['rows'] == 80
    nom = MecanumParams()
    noise = fp.dist_noise(rows(body, run_ms=200)[1])
    for row in rec:
        k = int(row[0])
        x = states[k - 1] if k else MOVING                          # state at the start of step k
        assert row[c['vx']] == x[3] and row[c['vy']] == x[4] and row[c['x']] == x[0]
        assert row[c['f_drag']] == (3.3 if k < 60 and math.hypot(x[3], x[4]) >= 1e-3 else 0.0), k
        assert row[c['f_dist']] == (fp.dist_value(rows(body, run_ms=200)[1], k, noise) if k >= 40 else 0.0), k
        assert row[c['V_batt']] == (10.0 if 50 <= k < 70 else nom.V_batt), k
        assert row[c['mask']] == float((k < 60) * 1 + (k >= 40) * 2 + (50 <= k < 70) * 4), k
    assert np.all(rec[:, c['b_fl']] == nom.motor_b[0]) and np.all(rec[:, c['mu_fl']] == nom.mu_hd[0])


def test_body_rows_alone_keep_the_nominal_parameter_object():
    r = rows('D1,body_drag,body,0,100,F=3.3\n')
    p = MecanumParams()
    var = fp.PlantVariant(r, p)
    assert var.at((0,), 5, np.zeros(10)) == (p, 0) and not var._cache


def test_hil_node_runs_body_faults(tmp_path):
    p = str(tmp_path / 'frames.npz')
    s = sched('D1,body_drag,body,100,300,F=3.3\n' + DIST.format(a=150, b=350), 400)
    plant = MecanumPlant(dt=0.001, seed=0)
    plant.reset(list(MOVING))
    node = HILNode(plant, config=HILConfig(simulate_hw=True, max_steps=400, period_us=1_000, frame_log=p, faults=s))
    node._spi.transfer = lambda mosi: DUTY
    node.run()
    assert isinstance(node.plant, fp.FaultPlant) and node.plant.inner is plant
    f = np.load(p)
    man = json.loads(str(f['manifest']))
    pr = man['faults']['plant_record']
    assert pr['rows'] == 250 and man['faults']['complete']
    rec = np.load(str(tmp_path / pr['file']))['record']
    assert list(rec[:, 0].astype(int)) == list(range(100, 350))
    assert np.any(rec[:, fp.RECORD_COLS.index('f_drag')] == 3.3)
    assert np.array_equal(f['mosi'], f['mosi_true'])                # body loads do not patch the frame


@pytest.mark.parametrize('kw,n_modes,lifted', [
    (dict(mock=True), 0, False),
    ({}, 1, False),
    ({}, 0, True),
    (dict(ground='rigid'), 0, False),
])
def test_run_hil_refuses_body_rows_off_the_traction_plant(tmp_path, kw, n_modes, lifted):
    import run_hil
    with pytest.raises(SystemExit):
        run_hil.load_faults(_args(tmp_path, 'D1,body_drag,body,1,9,F=3.3\n', **kw), n_modes,
                            HILConfig(simulate_hw=True), False, lifted)


def test_run_hil_accepts_body_rows(tmp_path):
    import run_hil
    s = run_hil.load_faults(_args(tmp_path, 'D1,body_drag,body,1,9,F=3.3\n' + DIST.format(a=1, b=9)),
                            0, HILConfig(simulate_hw=True), False, False)
    assert [r.fault for r in s.rows] == ['body_drag', 'load_dist']
