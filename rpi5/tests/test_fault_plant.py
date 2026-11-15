"""Fault injection batch 2, plant faults (fault_plant.py, fault_schedule.py; design 2026-11-15 sections 3, 4, 9):
schedule rows and their checks, the parameters of each step (PlantVariant), the wrapper of the Pi plant
(FaultPlant: nominal path unchanged, record, phase rule) and run_hil.py --faults refusals."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

import fault_plant as fp
import fault_schedule as fs
from hil_node import HILConfig, HILNode
from mecanum_plant import MecanumPlant
from params import MecanumParams
from protocol import FRAME_SIZE

HEAD = fs.MAGIC + '\nid,fault,target,start_ms,end_ms,params\n'
DUTY = bytes([0x30, 0x00] * 4) + bytes(FRAME_SIZE - 8)     # a constant forward duty from the "H7"
MOVING = [0, 0, 0, 0.2, 0, 0.0, 4, 4, 4, 4]                 # rolling forward from the first step


def rows(body, run_ms=1000, plant='traction', imu='plant'):
    r = fs.parse(HEAD + body)
    fs.check(r, run_ms, imu, plant)
    return r


def sched(body, run_ms=1000):
    return fs.Schedule(tuple(rows(body, run_ms)), 'x.csv', 'f' * 64)


# ── schedule ────────────────────────────────────────────────────────────────
def test_parse_plant_rows_and_defaults():
    r = rows('J1,wheel_jam,FL+RR,10,20,b=0.1\nJ2,wheel_jam,FR,10,20,tc=0.05\nB1,batt_sag,batt,0,900,v1=7.44;ramp_ms=700\n'
             'M1,mu_time,RL,30,40,mu=0.0545\nZ1,mu_zone,all,50,60,x0=1;x1=2;y0=-0.5;y1=0.5;mu=0.2\n')
    assert r[0].p('tc') == 0 and r[1].p('b') == 0                     # default 0 for the one left out
    assert r[0].wheels == (0, 3) and r[3].wheels == (2,) and r[4].wheels == (0, 1, 2, 3) and r[2].wheels == ()
    assert fs.has_plant_rows(r) and not fs.has_plant_rows(fs.parse(HEAD + 'E1,enc_loss,FL,1,2,\n'))


@pytest.mark.parametrize('body,plant', [
    ('J1,wheel_jam,FL,1,9,\n', 'traction'),                          # b and tc both 0
    ('J1,wheel_jam,FL,1,9,b=-0.1\n', 'traction'),
    ('J1,wheel_jam,FL,1,9,b=0.1;tc=-1\n', 'traction'),
    ('J1,wheel_jam,XX,1,9,b=0.1\n', 'traction'),
    ('J1,wheel_jam,batt,1,9,b=0.1\n', 'traction'),
    ('B1,batt_sag,batt,1,9,v1=0;ramp_ms=0\n', 'traction'),
    ('B1,batt_sag,batt,1,9,v1=7;ramp_ms=9\n', 'traction'),           # ramp longer than the window
    ('B1,batt_sag,FL,1,9,v1=7;ramp_ms=0\n', 'traction'),
    ('M1,mu_time,FL,1,9,mu=0\n', 'traction'),
    ('M1,mu_time,all,1,9,mu=0.2\n', 'traction'),                    # mu_time takes wheels
    ('Z1,mu_zone,all,1,9,x0=2;x1=1;y0=0;y1=1;mu=0.2\n', 'traction'),
    ('Z1,mu_zone,all,1,9,x0=0;x1=1;y0=1;y1=1;mu=0.2\n', 'traction'),
    ('Z1,mu_zone,FL,1,9,x0=0;x1=1;y0=0;y1=1;mu=0.2\n', 'traction'),
    ('J1,wheel_jam,FL,1,9,b=0.1\n', None),                           # no dynamic traction plant
    ('J1,wheel_jam,FL,1,9,b=0.1\nJ2,wheel_jam,FL+FR,5,12,tc=0.1\n', 'traction'),   # same wheel overlaps
    ('B1,batt_sag,batt,1,9,v1=7;ramp_ms=0\nB2,batt_sag,batt,8,12,v1=6;ramp_ms=0\n', 'traction'),
    ('M1,mu_time,RL,1,9,mu=0.2\nZ1,mu_zone,all,5,12,x0=0;x1=1;y0=0;y1=1;mu=0.2\n', 'traction'),
])
def test_check_refuses_plant_rows(body, plant):
    with pytest.raises(fs.ScheduleError):
        rows(body, plant=plant)


def test_check_allows_plant_overlaps_on_other_fields():
    rows('J1,wheel_jam,FL,1,9,b=0.1\nJ2,wheel_jam,FR,1,9,b=0.1\nM1,mu_time,FL,1,9,mu=0.2\nM2,mu_time,RR,1,9,mu=0.1\n'
         'B1,batt_sag,batt,1,9,v1=7;ramp_ms=8\nE1,enc_loss,FL,1,9,\nS1,gyro_scale,gz,1,9,s=0.2\n')


@pytest.mark.parametrize('body,edit', [
    ('M1,mu_time,FL,1,9,mu=0.5\n', None),                            # mu above the nominal 0.436
    ('Z1,mu_zone,all,1,9,x0=0;x1=1;y0=0;y1=1;mu=0.437\n', None),
    ('B1,batt_sag,batt,1,9,v1=12.5;ramp_ms=0\n', None),              # above the nominal 12.4 V
    ('J1,wheel_jam,FL,1,9,b=0.1\n', lambda p: setattr(p, 'ground_model', 'rigid')),
    ('J1,wheel_jam,FL,1,9,b=0.1\n', lambda p: p.update_physical(wheels_lifted=True)),
])
def test_check_params_against_nominal(body, edit):
    p = MecanumParams()
    if edit:
        edit(p)
    with pytest.raises(fs.ScheduleError):
        fp.check_params(rows(body), p)


def test_check_params_per_wheel_nominal_mu():
    p = MecanumParams()
    p.mu_hd = np.array([0.436, 0.3, 0.436, 0.436])
    fp.check_params(rows('M1,mu_time,FL,1,9,mu=0.4\n'), p)
    with pytest.raises(fs.ScheduleError):
        fp.check_params(rows('M1,mu_time,FR,1,9,mu=0.4\n'), p)


# ── parameters of a step ────────────────────────────────────────────────────
def test_variant_values():
    base = MecanumParams()
    b0, tc0, ts0, mu0, v0 = (np.array(base.motor_b), np.array(base.motor_Tc), np.array(base.motor_Ts),
                             np.array(np.broadcast_to(base.mu_hd, (4,))), base.V_batt)
    r = rows('J1,wheel_jam,FL+RR,0,100,b=0.1;tc=0.05\nB1,batt_sag,batt,0,100,v1=7.4;ramp_ms=50\n'
             'M1,mu_time,FR,0,100,mu=0.05\n')
    var = fp.PlantVariant(r, base)
    assert var.at((), 3, MOVING) == (base, 0)                       # nothing on: the nominal object itself
    p, _ = var.at((0, 1, 2), 0, MOVING)
    assert p is not base
    assert np.array_equal(p.motor_b, b0 + [0.1, 0, 0, 0.1]) and np.array_equal(p.motor_Tc, tc0 + [0.05, 0, 0, 0.05])
    assert np.array_equal(p.motor_Ts, ts0 + [0.05, 0, 0, 0.05]) and np.array_equal(p.mu_hd, [mu0[0], 0.05, *mu0[2:]])
    assert p.V_batt == v0                                            # ramp start: nominal
    assert var.at((0, 1, 2), 25, MOVING)[0].V_batt == pytest.approx(v0 + (7.4 - v0) * 0.5, abs=1e-12)
    assert var.at((0, 1, 2), 50, MOVING)[0].V_batt == 7.4 and var.at((0, 1, 2), 99, MOVING)[0].V_batt == 7.4
    # the nominal object is never changed
    assert np.array_equal(base.motor_b, b0) and base.V_batt == v0 and np.array_equal(base.mu_hd, mu0)


def test_batt_step_when_ramp_is_zero():
    r = rows('B1,batt_sag,batt,10,100,v1=6;ramp_ms=0\n')
    assert fp.PlantVariant(r, MecanumParams()).at((0,), 10, MOVING)[0].V_batt == 6.0


def test_zone_per_wheel_hub_and_heading():
    base = MecanumParams()                                           # lx = ly = 0.12
    r = rows('Z1,mu_zone,all,0,100,x0=1;x1=2;y0=-1;y1=1;mu=0.05\n')
    var = fp.PlantVariant(r, base)
    # centre at x = 0.9: the front hubs (x = 1.02) are in, the rear ones (0.78) out
    p, zm = var.at((0,), 0, [0.9, 0, 0])
    assert zm == 0b0011 and np.array_equal(p.mu_hd[:2], [0.05, 0.05]) and p.mu_hd[2] == base.mu_hd[2]
    # heading +90 deg: body x = world y, so FL (+lx, +ly) -> world (x - ly, y + lx): left hubs are behind
    _, zm = var.at((0,), 0, [1.05, 0, np.pi / 2])
    assert zm == 0b1010                                             # FR (x 1.17) and RR (1.17) in, FL, RL (0.93) out
    assert var.at((0,), 0, [0.0, 0, 0])[1] == 0                     # all out: nominal mu
    assert fp.zone_mask(r[0], [1.88, 0, 0], base) == 0b1100         # half open: front hubs at 2.00 are out


# ── the Pi wrapper ──────────────────────────────────────────────────────────
class _Run:
    """A FramePatcher driven as hil_node does (patch, then plant.step), with a FaultPlant around a MecanumPlant."""

    def __init__(self, body, t0=0, stop=None, run_ms=1000, path=None, state=MOVING):
        self.patcher = fs.FramePatcher(sched(body, run_ms))
        self.inner = MecanumPlant(dt=0.001, seed=0)
        self.inner.reset(list(state))
        self.plant = fp.FaultPlant(self.inner, self.patcher, path)
        self.patcher.plant_hook = self.plant
        self.t0, self.stop = t0, stop

    def go(self, n, duty=(0.3, 0.3, 0.3, 0.3)):
        out = []
        for k in range(n):
            if k == self.t0:
                self.patcher.start(k)
            if self.stop is not None and k == self.stop:
                self.patcher.stop(k)
            self.patcher.patch(k, bytes(FRAME_SIZE))
            out.append(self.plant.step(list(duty)))
        return np.array(out)


def plain(n, duty=(0.3, 0.3, 0.3, 0.3), state=MOVING):
    p = MecanumPlant(dt=0.001, seed=0)
    p.reset(list(state))
    return np.array([p.step(list(duty)) for _ in range(n)])


def test_no_row_on_is_bit_identical_to_the_plain_plant():
    r = _Run('J1,wheel_jam,FL,1500,1600,b=0.1\n', run_ms=2000)
    assert np.array_equal(r.go(1500), plain(1500))                 # before the window: same bits
    assert r.plant._n == 0


def test_wheel_jam_slows_only_its_wheel():
    s = _Run('J1,wheel_jam,FL,0,1000,b=0.1\n').go(800)               # motor_polarity -1: wheels turn < 0
    s0 = plain(800)
    assert abs(s[-1, 6]) < 0.5 * abs(s0[-1, 6])                      # FL far slower
    assert np.all(np.abs(s[-1, 7:10] - s0[-1, 7:10]) < 0.35 * np.abs(s0[-1, 7:10]))


def test_batt_sag_lowers_speed_and_mu_time_makes_wheels_slip():
    s = _Run('B1,batt_sag,batt,0,1000,v1=6;ramp_ms=0\n').go(800)
    s0 = plain(800)
    assert np.all(np.abs(s[-1, 6:10]) < 0.8 * np.abs(s0[-1, 6:10]))
    m = _Run('M1,mu_time,FL+FR+RL+RR,0,1000,mu=0.02\n').go(400, duty=(1, 1, 1, 1))
    m0 = plain(400, duty=(1, 1, 1, 1))
    # body slower, wheels spin faster (sizes: motor_polarity -1)
    assert abs(m[-1, 3]) < abs(m0[-1, 3]) and np.mean(np.abs(m[-1, 6:10])) > np.mean(np.abs(m0[-1, 6:10]))


def test_record_window_phase_and_manifest(tmp_path):
    path = str(tmp_path / 'r_plant_faults.npz')
    r = _Run('J1,wheel_jam,FL,10,30,b=0.1\nB1,batt_sag,batt,20,60,v1=10;ramp_ms=20\n', t0=5, stop=50, path=path)
    r.go(100)
    summ = r.patcher.summary(last_step=99)
    pr = summ['plant_record']
    f = np.load(path)
    rec = f['record']
    steps = rec[:, 0].astype(int)
    # rows on at relative k in [10, 60) but the run left RUN at absolute 50 (relative 45): cut there
    assert list(steps) == list(range(15, 50)) and pr['rows'] == 35 and not pr['overflow']
    assert pr['file'] == os.path.basename(path) and len(pr['sha256']) == 64
    assert json.loads(str(f['cols'])) == list(fp.RECORD_COLS)
    mask = rec[:, 1].astype(int)
    assert set(mask[steps < 25]) == {1} and set(mask[(steps >= 25) & (steps < 35)]) == {3} and set(mask[steps >= 35]) == {2}
    v = rec[:, 2]
    assert v[steps == 25][0] == MecanumParams().V_batt and v[steps == 45][0] == 10.0
    assert any(e[3] == 'cut' for e in summ['events'])
    assert r.patcher.summary(last_step=99)['plant_record'] is pr                   # saved once


def test_attach_only_with_plant_rows(tmp_path):
    p = MecanumPlant(dt=0.001)
    sig = fs.FramePatcher(fs.Schedule(tuple(fs.parse(HEAD + 'E1,enc_loss,FL,1,2,\n')), 'x', 'y'))
    assert fp.attach(p, sig, 'a.npz') is p and fp.attach(p, None, None) is p
    pl = fs.FramePatcher(sched('J1,wheel_jam,FL,1,2,b=0.1\n'))
    w = fp.attach(p, pl, str(tmp_path / 'a.npz'))
    assert isinstance(w, fp.FaultPlant) and pl.plant_hook is w and w._path.endswith('a_plant_faults.npz')
    from plant_interface import MockPlant
    with pytest.raises(fs.ScheduleError):
        fp.attach(MockPlant(), fs.FramePatcher(sched('J1,wheel_jam,FL,1,2,b=0.1\n')), 'a.npz')


def test_hil_node_runs_plant_faults_and_writes_the_record(tmp_path):
    p = str(tmp_path / 'frames.npz')
    s = sched('J1,wheel_jam,FL,100,300,b=0.1\nM1,mu_time,RR,150,250,mu=0.1\n', 400)
    plant = MecanumPlant(dt=0.001, seed=0)
    plant.reset(list(MOVING))
    node = HILNode(plant, config=HILConfig(simulate_hw=True, max_steps=400, period_us=1_000, frame_log=p, faults=s))
    node._spi.transfer = lambda mosi: DUTY
    node.run()
    assert isinstance(node.plant, fp.FaultPlant) and node.plant.inner is plant
    f = np.load(p)
    man = json.loads(str(f['manifest']))
    assert man['plant'] == 'FaultPlant'
    pr = man['faults']['plant_record']
    assert pr['rows'] == 200 and pr['file'] == 'frames_plant_faults.npz'
    rec = np.load(str(tmp_path / pr['file']))['record']
    assert list(rec[:, 0].astype(int)) == list(range(100, 300))
    assert np.array_equal(f['mosi'], f['mosi_true'])                # plant faults do not patch the frame
    assert man['faults']['complete']


def test_hil_node_refuses_plant_rows_on_a_mock_plant(tmp_path):
    from plant_interface import MockPlant
    with pytest.raises(fs.ScheduleError):
        HILNode(MockPlant(), config=HILConfig(simulate_hw=True, max_steps=10, frame_log=str(tmp_path / 'a.npz'),
                                             faults=sched('J1,wheel_jam,FL,1,2,b=0.1\n')))


# ── run_hil.py --faults ─────────────────────────────────────────────────────
def _args(tmp_path, body, **kw):
    csv_ = tmp_path / 's.csv'
    csv_.write_text(HEAD + body)
    a = dict(faults=str(csv_), frame_log='f.npz', step_log='s.csv', mock=False, steps=1000, ground=None, mu=None)
    a.update(kw)
    return SimpleNamespace(**a)


@pytest.mark.parametrize('kw,n_modes,lifted', [
    (dict(mock=True), 0, False),
    ({}, 1, False),                                                   # --fixed-omega and co.
    ({}, 0, True),                                                    # A-K* (wheels lifted)
    (dict(ground='rigid'), 0, False),
    (dict(mu=0.05), 0, False),                                        # nominal 0.05 < the row's 0.1
])
def test_run_hil_refuses_plant_rows(tmp_path, kw, n_modes, lifted):
    import run_hil
    cfg = HILConfig(simulate_hw=True)
    with pytest.raises(SystemExit):
        run_hil.load_faults(_args(tmp_path, 'M1,mu_time,FL,1,9,mu=0.1\n', **kw), n_modes, cfg, False, lifted)


def test_run_hil_accepts_plant_rows(tmp_path):
    import run_hil
    s = run_hil.load_faults(_args(tmp_path, 'M1,mu_time,FL,1,9,mu=0.1\nJ1,wheel_jam,FR,1,9,b=0.1\n', mu=0.3),
                            0, HILConfig(simulate_hw=True), False, False)
    assert fs.has_plant_rows(s.rows)
