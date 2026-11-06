"""Tests of fwsim/s03_doe.py (spec S03 v6) and o1_feasibility.evaluate_v7 (O1 v7 section 5.9,
S03 v6 section 4). Synthetic logs and responses with a known answer; the last tests need
fwsim/build/fwsim.dll (firmware with pos_ctrl.c). Run: python -m pytest fwsim/tests -q"""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import fwsim  # noqa: E402
import o1_feasibility as o1  # noqa: E402
import s03_doe as s3  # noqa: E402

_DLL = os.path.join(fwsim.HERE, 'build', 'fwsim.dll')


@pytest.fixture(scope='module')
def path():
    return o1.Path8()


def _ideal(sch, dt=1e-4):
    """Ideal car from the origin (heading 45 deg on the floor), plant frame, 100 Hz log."""
    n = int(round((sch.Tcmd + o1.T_LATE) / dt))
    x = y = th = 0.0
    t_, pose, vel = [], [], []
    for k in range(n + 1):
        t = k * dt
        vx, vy, wz = sch.setpoint(t)
        if k % 100 == 0 and k:
            t_.append(t)
            pose.append((x, y, th))
            vel.append((vx, vy, wz))
        x += vx * math.cos(th) * dt
        y += vx * math.sin(th) * dt
        th += wz * dt
    m = len(t_)
    return dict(t=np.array(t_), pose=np.array(pose), v_true=np.array(vel), pwm=np.zeros((m, 4)),
                duty=np.zeros((m, 4)))


def _plant_xy(fx, fy):
    """Floor point -> plant frame (inverse of the 45 deg start rotation of evaluate)."""
    c, s = math.cos(o1.HEAD0), math.sin(o1.HEAD0)
    return c * fx + s * fy, -s * fx + c * fy


class _Later:
    """A schedule whose T_cmd is later than the motion: tests condition 5 (t >= T_cmd)."""

    def __init__(self, sch, extra):
        self.__dict__.update(sch.__dict__)
        self.Tcmd = sch.Tcmd + extra
        self.tau = sch.tau


# ── evaluate_v7 ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('v', [0.2, 0.8])
def test_v7_ideal_car(path, v):
    sch = o1.Schedule(v)
    res, rows = o1.evaluate_v7(_ideal(sch), path, sch)
    assert res['completed'] == 'true' and res['status'] == 'ok' and res['track_err_frames'] == 0
    assert res['e_perp_max'] < 0.002 and res['e_trk_max'] < 0.01 and res['e_theta_cmd_max_deg'] < 0.5
    assert res['t_end'] == pytest.approx(sch.Tcmd + o1.T_D, abs=0.02)
    assert res['e_perp_global_max'] <= res['e_perp_max'] + 1e-9


def test_v7_condition5_waits_for_the_command_end(path):
    """The car is still at the origin 3 s before T_cmd: v5 ended there, v7 waits for T_cmd (O6-3)."""
    sch = o1.Schedule(0.8)
    log = _ideal(sch)
    later = _Later(sch, 3.0)
    log = {k: np.concatenate([v, np.repeat(v[-1:], 300, axis=0)]) for k, v in log.items() if k != 't'} | \
        dict(t=np.concatenate([log['t'], log['t'][-1] + 0.01 * np.arange(1, 301)]))
    res, _ = o1.evaluate_v7(log, path, later)
    assert res['completed'] == 'true'
    assert res['t_end'] == pytest.approx(later.Tcmd, abs=0.02)
    old, _ = o1.evaluate(log, path, later)
    assert old['t_end'] < later.Tcmd - 1.0


def test_v7_turning_in_place_is_not_still(path):
    """Condition 3 also on |wz| < omega_d = v_d / (lx + ly) (O6-18)."""
    sch = o1.Schedule(0.8)
    log = _ideal(sch)
    log['v_true'][log['t'] > sch.Tcmd, 2] = 0.1
    res, _ = o1.evaluate_v7(log, path, sch)
    assert res['completed'] == 'false' and not res['cond2_5_ever']


def test_v7_track_error_before_progress_is_inconclusive(path):
    """One frame jumps 1 m ahead mid-run (upper end of the window): progress not yet reached,
    conditions 2-5 hold at the end -> inconclusive, status track_err (S03 4, S3-V-1)."""
    sch = o1.Schedule(0.5)
    log = _ideal(sch)
    k = int(np.searchsorted(log['t'], 4.0))
    s_k = path.s_of_tau(sch.tau(log['t'][k])[0])
    j = int(np.searchsorted(path.s, s_k + 1.0))
    log['pose'][k, :2] = _plant_xy(path.x[j], path.y[j])
    res, _ = o1.evaluate_v7(log, path, sch)
    assert res['status'] == 'track_err' and res['completed'] == 'inconclusive'
    assert res['t_first_track_err'] == pytest.approx(log['t'][k])


def test_v7_track_error_after_progress_keeps_completion(path):
    """A frame 0.5 m back on the lane after progress reached S - D_d (B8b-1): still 'true'."""
    sch = o1.Schedule(0.5)
    log = _ideal(sch)
    k = int(np.searchsorted(log['t'], sch.Tcmd + 0.3))
    j = int(np.searchsorted(path.s, path.S - 0.5))
    log['pose'][k, :2] = _plant_xy(path.x[j], path.y[j])
    res, _ = o1.evaluate_v7(log, path, sch)
    assert res['status'] == 'track_err' and res['completed'] == 'true'
    assert res['t_progress'] < res['t_first_track_err']


def test_v7_track_error_and_never_at_origin_is_false(path):
    """Car parked 0.5 m from the origin, one jump frame: conditions 2-5 never hold -> 'false'."""
    sch = o1.Schedule(0.8)
    log = _ideal(sch)
    log['pose'][:] = (0.5, 0.0, 0.0)
    log['v_true'][:] = 0.0
    k = 100
    log['pose'][k, :2] = _plant_xy(path.x[200000], path.y[200000])
    res, _ = o1.evaluate_v7(log, path, sch)
    assert res['status'] == 'track_err' and res['completed'] == 'false'


def test_v7_lag_sign_and_window_of_y3_y4(path):
    """e_s > 0 when the car is behind the command (O1 v7 S2); Y3, Y4 only on [0, t_end]."""
    sch = o1.Schedule(0.8)
    log = _ideal(sch)
    after = log['t'] > sch.Tcmd + o1.T_D + 0.05
    log['duty'][after, 0] = np.where(np.arange(after.sum()) % 2, 0.5, -0.5)   # duty chatter after t_end
    log['pwm'][:, 1] = 1000                                                      # one wheel saturated throughout
    res, _ = o1.evaluate_v7(log, path, sch)
    assert res['tv_duty'] == 0.0
    assert res['sat_frac'] == pytest.approx(0.25) and res['sat_any_frac'] == pytest.approx(1.0)
    still = _ideal(sch)
    still['pose'][:] = 0.0
    still['v_true'][:] = 0.0
    _, rows = o1.evaluate_v7(still, path, sch)
    assert rows[100]['e_s'] > 0 and rows[100]['s_cmd'] - rows[100]['s_car'] > 0


def test_v7_censored_and_measurement_error(path):
    sch = o1.Schedule(0.8)
    log = _ideal(sch)
    log['pose'][:, 0] += 0.7                     # whole run 0.7 m off: censored at 0.5 m
    res, _ = o1.evaluate_v7(log, path, sch)
    assert res['e_perp_max'] > 0.5 and res['e_perp_max_cens'] == o1.C_CENS
    assert math.isnan(res['e_meas_perp_max'])    # no firmware pose columns


# ── S03 pieces ───────────────────────────────────────────────────────────────────────

def test_sweep_grid_rules():
    g = s3.sweep_grid('kp_xy', 1.5, 0.0, 6.0)               # a = 0.15, b = 15, plus 0
    assert g[0] == 0.0 and 1.5 in g and g == sorted(g)
    assert min(x for x in g if x > 0) == pytest.approx(0.15) and max(g) == pytest.approx(15.0)
    assert len(g) == 12                                     # 11 log points (1.5 is one of them) + 0
    g = s3.sweep_grid('omega_o_xy', 30.0, 5.0, 48.0)        # a = min(3, 5), b = min(max(300, 48), 48)
    assert 0.0 not in g and 30.0 in g and max(g) == 48.0 and g[0] == pytest.approx(3.0) and len(g) == 12
    g = s3.sweep_grid('omega_c_xy', 8.0, 1.0, 25.0)         # a = min(0.8, 1), b = max(80, 25)
    assert g[0] == pytest.approx(0.8) and max(g) == pytest.approx(80.0) and 8.0 in g


def test_level_rule():
    grid = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert s3.levels_from_sweep(grid, 2.0, [False, True, True, True, False, True]) == (1.0, 3.0)
    assert s3.levels_from_sweep(grid, 2.0, [True] * 6) == (0.0, 5.0)
    assert s3.levels_from_sweep(grid, 2.0, [True, False, True, False, True, True]) == (2.0, 2.0)  # L = H
    with pytest.raises(s3.S03Stop, match=r'3\.1 \(i\)'):
        s3.levels_from_sweep(grid, 2.0, [True, True, False, True, True, True])


def test_scale_and_decode():
    assert s3.scale_of(1.0, 4.0) == ('log', 2.0)
    assert s3.scale_of(0.0, 3.0) == ('lin', 1.5) and s3.scale_of(1.0, 3.0) == ('lin', 2.0)
    for L, H in ((1.0, 16.0), (0.0, 3.0)):
        sc = s3.scale_of(L, H)[0]
        assert s3.decode(-1, L, H, sc) == pytest.approx(L) and s3.decode(1, L, H, sc) == pytest.approx(H)
        assert s3.decode(0, L, H, sc) == pytest.approx(s3.scale_of(L, H)[1])
    assert s3.decode(0.5, 1.0, 16.0, 'log') == pytest.approx(8.0)     # sqrt(16) * 4^0.5


def test_design_is_full_factorial():
    pts = s3.design_points()
    codes = np.array([c for _, c in pts])
    assert [p for p, _ in pts] == list(range(1, 129)) and len({tuple(c) for c in codes}) == 128
    assert np.all(codes.sum(axis=0) == 0)
    for i in range(7):
        for j in range(i + 1, 7):
            assert (codes[:, i] * codes[:, j]).sum() == 0
    assert codes[0, 0] == -1 and codes[1, 0] == 1 and codes[1, 1] == -1     # X1 changes fastest
    inner = np.array([c for _, c in s3.design_points(0.5, 129)])
    assert set(np.unique(inner)) == {-0.5, 0.5}


def test_effects_known_model():
    codes = np.array([c for _, c in s3.design_points()])
    x = codes
    y = 2.0 + 0.5 * x[:, 0] + 0.25 * x[:, 0] * x[:, 1] - 0.1 * x[:, 2] * x[:, 3] * x[:, 4]
    e = s3.effects(codes, y)
    assert len(e) == 127
    assert e['X1'] == pytest.approx(1.0) and e['X1:X2'] == pytest.approx(0.5)
    assert e['X3:X4:X5'] == pytest.approx(-0.2)
    assert all(abs(v) < 1e-12 for k, v in e.items() if k not in ('X1', 'X1:X2', 'X3:X4:X5'))
    half = codes * 0.5                                    # inner box: effect at its own width (S3-R-3)
    e2 = s3.effects(half, 2.0 + 0.5 * half[:, 0])
    assert e2['X1'] == pytest.approx(0.5)


def _runs(model, mu_levels=(0.4, 0.5, 0.6), speeds=(0.5,)):
    pts = s3.design_points()
    runs = {}
    for pid, c in [(0, (0.0,) * 7)] + pts:
        for mu in mu_levels:
            for v in speeds:
                runs[(pid, mu, v)] = model(np.array(c), mu)
    return pts, runs


def _row(e, tv=1.0, sat=0.0, completed='true', status='ok'):
    return dict(e_perp_max=e, e_perp_max_cens=min(e, 0.5), tv_duty=tv, sat_frac=sat, completed=completed,
                status=status)


def test_analyse_worst_range_and_thresholds():
    """Y2 = 0.05 + 0.01 X1 + 0.02 mu X2: effect X1 0.02 (significant at 0.01), X2 2*0.02*mu;
    worst over mu takes mu = 0.6; Y3 threshold 20 % of the reference."""
    def model(c, mu):
        return _row(0.05 + 0.01 * c[0] + 0.02 * mu * c[1], tv=1.0 + 0.1 * c[2],
                    completed='false' if c[3] > 0 and c[4] > 0 else 'true')
    pts, runs = _runs(model)
    eff, curv, ver, fails = s3.analyse('PID', 'outer', pts, runs, [0.4, 0.5, 0.6], [0.5], {('PID', 0.5): 1.0})
    get = {(r['response'], r['mu'], r['term']): r for r in eff}
    assert get[('e_perp_max', '0.5', 'X1')]['effect'] == pytest.approx(0.02)
    assert get[('e_perp_max', '0.5', 'X1')]['significant']
    # worst: X2 = +1 is worst at mu 0.6 (+0.012), X2 = -1 at mu 0.4 (-0.008): effect 0.020
    assert get[('e_perp_max', 'worst', 'X2')]['effect'] == pytest.approx(0.012 + 0.008)
    # range over mu: 0.004 for both levels of X2, so its effect is 0
    assert get[('e_perp_max', 'range', 'X2')]['effect'] == pytest.approx(0.0, abs=1e-12)
    assert get[('e_perp_max', 'range', 'X3')]['effect'] == pytest.approx(0.0, abs=1e-12)
    r3 = get[('tv_duty', '0.5', 'X3')]
    assert r3['threshold'] == pytest.approx(0.2) and r3['effect'] == pytest.approx(0.2) and not r3['significant']
    assert all(v['main'] == 'as_measured' for v in ver)
    nc = {(f['factors'], f['levels'], f['mu']): f['count'] for f in fails if f['kind'] == 'not_completed'}
    assert nc[('X4:X5', '+,+', 0.5)] == 32 and nc[('X4:X5', '-,+', 0.5)] == 0 and nc[('X4', '+', 0.5)] == 32
    c = {(r['response'], r['mu']): r for r in curv}
    assert c[('e_perp_max', '0.5')]['diff'] == pytest.approx(0.0)


def test_analyse_censoring_changes_the_main_version():
    """A lost run (Y2 3 m) at one corner flips a significant set between Y2 and censored Y2."""
    def model(c, mu):
        big = c[0] > 0 and c[1] > 0 and c[2] > 0 and c[3] > 0 and c[4] > 0 and c[5] > 0 and c[6] > 0
        return _row(3.0 if big else 0.05, completed='false' if big else 'true')
    pts, runs = _runs(model, mu_levels=(0.5,))
    _, _, ver, _ = s3.analyse('ADRC', 'outer', pts, runs, [0.5], [0.5], {('ADRC', 0.5): 1.0})
    assert any(v['main'] == 'both' for v in ver)


def test_analyse_counts_track_errors_and_inconclusive():
    def model(c, mu):
        if c[0] > 0 and c[1] > 0:
            return _row(0.05, completed='inconclusive', status='track_err')
        return _row(0.05)
    pts, runs = _runs(model, mu_levels=(0.5,))
    eff, _, _, fails = s3.analyse('PID', 'outer', pts, runs, [0.5], [0.5], {('PID', 0.5): 1.0})
    r = next(x for x in eff if x['mu'] == '0.5' and x['term'] == 'X1')
    assert r['n_track_err'] == 32 and r['n_inconclusive'] == 32
    inc = {(f['factors'], f['levels']): f['count'] for f in fails if f['kind'] == 'inconclusive'}
    assert inc[('X1:X2', '+,+')] == 32


def test_log_digest_detects_one_bit():
    a = dict(x=np.array([1.0, 2.0]))
    b = dict(x=np.array([1.0, np.nextafter(2.0, 3.0)]))
    assert s3.log_digest(a) == s3.log_digest(dict(x=np.array([1.0, 2.0])))
    assert s3.log_digest(a) != s3.log_digest(b)


# ── with the DLL ─────────────────────────────────────────────────────────────────────

needs_dll = pytest.mark.skipif(not os.path.isfile(_DLL), reason='fwsim.dll not built')


@needs_dll
@pytest.mark.parametrize('ctrl', ['PID', 'ADRC'])
def test_defaults_and_gain_setting(ctrl):
    fw = fwsim.FirmwareSim(_DLL)
    if not fw.pos_cols:
        pytest.skip('DLL without pos_ctrl.c')
    d = s3.defaults(fw, ctrl)
    want = (dict(kp_xy=1.5, ki_xy=0.05, kp_wz=1.0, ki_wz=0.05) if ctrl == 'PID' else
            dict(omega_c_xy=8.0, omega_o_xy=30.0, omega_c_wz=8.0, omega_o_wz=30.0))
    want.update(pos_kp=6.0, pos_ki=3.0, k_psi=1.0)          # S03 v6 section 2.1 table
    assert d == pytest.approx(want)
    kw = s3.sim_kwargs(fw, ctrl, dict(d, **{s3.FACTORS[ctrl][2]: 2.5}))
    inner = kw['pid'] if ctrl == 'PID' else kw['adrc']
    assert inner[2][0] == 2.5 and inner[0][0] == pytest.approx(want[s3.FACTORS[ctrl][0]])
    assert kw['fw_params'] == pytest.approx(dict(pos_kp=6.0, pos_ki=3.0, k_psi=1.0))


@needs_dll
def test_run_one_and_new_process_rerun_are_identical(path, tmp_path):
    fw = fwsim.FirmwareSim(_DLL)
    if not fw.pos_cols:
        pytest.skip('DLL without pos_ctrl.c')
    g = s3.defaults(fw, 'PID')
    row, log = s3.run_one(fw, path, 'PID', g, 0.8, 0.5)
    assert set(row) >= {'status', 'completed', 'e_perp_max', 'tv_duty', 'sat_frac', 'pos_reset_mid'}
    assert row['pos_reset_mid'] == 0
    again = s3.run_in_new_process(_DLL, [('PID', g, 0.8, 0.5)], str(tmp_path))[0]
    assert again['digest'] == s3.log_digest(log) and s3.row_text(again['row']) == s3.row_text(row)


@needs_dll
def test_main_pipeline_with_a_fake_simulator(tmp_path, monkeypatch):
    """main() end to end with an instant fake run (responses from the codes): every output file,
    sealed files in sealed/, meta.yaml, exit code 0. The DLL is only used for the defaults."""
    calls = []

    def fake_safe_run(fw, path, ctrl, g, v, mu):
        x = np.array([g[n] for n in s3.FACTORS[ctrl]])
        calls.append((ctrl, v, mu))
        log = dict(t=np.arange(3.0), duty=np.zeros((3, 4)) + x.sum())
        return dict(status='ok', completed='true', e_perp_max=0.01 + 1e-3 * float(x[0]), e_perp_max_cens=0.01,
                    e_perp_global_max=0.01, tv_duty=1.0 + float(x[1]), sat_frac=0.0, sat_any_frac=0.0,
                    e_theta_lane_max_deg=1.0, e_theta_cmd_max_deg=1.0, ds_absmax=0.0, e_trk_max=0.0,
                    t_end=1.0, e_s_absmax=0.0, track_err_frames=0, cond2_5_ever=True, pos_reset_mid=0), log

    def fake_new_process(dll, runs, work):
        return [dict(row=r, digest=s3.log_digest(l)) for r, l in (fake_safe_run(None, None, *j) for j in runs)]

    monkeypatch.setattr(s3, 'safe_run', fake_safe_run)
    monkeypatch.setattr(s3, 'run_in_new_process', fake_new_process)
    out = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', ['s03_doe.py', '--out', str(out), '--mu', '0.5', '--trial'])
    assert s3.main() == 0
    for f in ('sweep.csv', 'levels.csv', 'design.csv', 'determinism_check.csv', 'meta.yaml',
              'sealed/runs.csv', 'sealed/effects.csv', 'sealed/fail_counts.csv', 'sealed/notes.json'):
        assert (out / f).is_file(), f
    import csv as _csv
    runs = list(_csv.DictReader(open(out / 'sealed' / 'runs.csv')))
    assert len(runs) == 2 * 129 * 3                      # 2 controllers x (128 + centre) x 3 speeds x 1 mu
    det = list(_csv.DictReader(open(out / 'determinism_check.csv')))
    assert len(det) == 2 * 4 * 3 and all(r['log_arrays_equal'] == 'true' for r in det)
    lv = list(_csv.DictReader(open(out / 'levels.csv')))
    assert len(lv) == 14
    meta = (out / 'meta.yaml').read_text(encoding='utf-8')
    assert 'trial: true' in meta and 'sealed/runs.csv' in meta and 'src' in meta


def test_analyse_writes_the_box_notes():
    pts, runs = _runs(lambda c, mu: _row(0.05), mu_levels=(0.5,))
    eff, _, _, _ = s3.analyse('PID', 'outer', pts, runs, [0.5], [0.5], {('PID', 0.5): 1.0},
                              {('outer', 0.5): 'most corners not completed'})
    assert all(r['note'] == 'most corners not completed' for r in eff)
    eff, _, _, _ = s3.analyse('PID', 'outer', pts, runs, [0.5], [0.5], {('PID', 0.5): 1.0})
    assert all(r['note'] == '' for r in eff)
