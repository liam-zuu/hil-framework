"""Fault injection batch 1 in fwsim (fault_run.py, L82 Q1 = F2): no row = bit identical to fwsim.simulate,
wrapper removed (also on error), each fault reaches the firmware as on the Pi, the call order inside one
simulate step that the wrapper relies on.
Run: python -m pytest fwsim/tests (needs fwsim/build/fwsim.dll: python fwsim/build_fwsim.py)"""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'rpi5'))
import fwsim                                          # noqa: E402
import fault_run                                      # noqa: E402
import fault_schedule as fs                           # noqa: E402
from params import MecanumParams                      # noqa: E402
from protocol import _to_int16                        # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.isfile(os.path.join(fwsim.HERE, 'build', 'fwsim.dll')),
                                reason='fwsim.dll not built')
HEAD = fs.MAGIC + '\nid,fault,target,start_ms,end_ms,params\n'
IMU = dict(rate_hz=20.0, noise='replay', noise_seed=0)
GZR = fault_run.GZ_RANGE


@pytest.fixture(scope='module')
def fw():
    return fwsim.FirmwareSim()


def sched(body, run_ms=6000):
    rows = fs.parse(HEAD + body)
    fs.check(rows, run_ms, 'plant')
    return fs.Schedule(tuple(rows), 'x.csv', 'abc')


def _sp(t):
    return (0.2, 0.0, 1.0 * math.sin(2 * math.pi * t / 4.0))


def run(fw, s, duration=6.0, mode='PID', imu=IMU):
    return fault_run.FwsimFaults(s).run(fw, _sp, duration, MecanumParams(), imu=dict(imu), mode=mode)


def test_no_row_is_bit_identical_and_wrapper_removed(fw):
    a = fwsim.simulate(fw, _sp, 6.0, MecanumParams(), imu=dict(IMU), mode='PID')
    b = run(fw, fs.Schedule(()))
    for k in a:
        assert np.array_equal(a[k], b[k], equal_nan=True), k
    assert fwsim.plant_step is fault_run._REAL_PLANT_STEP
    assert b['faults']['complete'] and b['faults']['events'] == []


def test_wrapper_removed_after_an_error(fw, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError('x')
    monkeypatch.setattr(fwsim, 'simulate', boom)
    with pytest.raises(RuntimeError):
        run(fw, sched('E1,enc_loss,FL,100,200,\n'))
    assert fwsim.plant_step is fault_run._REAL_PLANT_STEP


def test_refuses_no_imu_injection_and_late_window(fw):
    with pytest.raises(ValueError):
        fault_run.FwsimFaults(fs.Schedule(())).run(fw, _sp, 1.0, MecanumParams(), imu=None)
    with pytest.raises(ValueError):
        run(fw, fs.Schedule(()), imu=dict(IMU, injection=object()))
    with pytest.raises(ValueError):
        run(fw, sched('E1,enc_loss,FL,100,5000,\n'), duration=4.0)


def test_call_order_inside_one_step(fw, monkeypatch):
    """The wrapper assumes: at t = k ms the IMU report, then fw.tick, then plant_step(x_k). If fwsim.py ever
    changes this order, the meaning of the faults shifts by one step: this test then fails."""
    calls = []
    real = fwsim.plant_step

    def plant(x, u, p, dt, rng):
        calls.append(('plant',))
        return real(x, u, p, dt, rng)

    class Rec:
        def __init__(self, f):
            self.f = f

        def __getattr__(self, n):
            return getattr(self.f, n)

        def imu_report(self, kind, t_us, v):
            calls.append(('report', t_us))
            return self.f.imu_report(kind, t_us, v)

        def tick(self, t_us, counts, sp):
            calls.append(('tick', t_us))
            return self.f.tick(t_us, counts, sp)

    monkeypatch.setattr(fwsim, 'plant_step', plant)
    fwsim.simulate(Rec(fw), _sp, 0.2, MecanumParams(), imu=dict(IMU), mode='PID')
    i = calls.index(('report', 50_000))
    assert calls[i:i + 5] == [('report', 50_000)] * 3 + [('tick', 50_000), ('plant',)]
    # exactly one plant step between two ticks 10 ms apart: 10 steps
    t1, t2 = calls.index(('tick', 50_000)), calls.index(('tick', 60_000))
    assert sum(c == ('plant',) for c in calls[t1:t2]) == 10


def test_enc_loss_freezes_only_that_wheel(fw):
    log = run(fw, sched('E1,enc_loss,FR,3000,4000,\n'))
    t, c = log['t'], log['count']
    w = (t > 3.011) & (t <= 4.0)
    assert np.ptp(c[w, 1]) == 0 and np.ptp(c[w, 0]) > 50 and np.ptp(c[w, 2]) > 50
    ft = log['fault_tick']
    before = ft[:, 0] < 3.0
    assert np.array_equal(ft[before][:, 2:6], ft[before][:, 6:10])
    assert np.ptp(ft[w, 3]) > 50                                   # the true wheel kept turning
    on = (ft[:, 0] >= 3.0) & (ft[:, 0] < 4.0)                       # half open [3000, 4000) ms
    assert np.all(ft[on, 1] == 2) and np.all(ft[~on, 1] == 0)
    # a window edge between two reports: the tick mask still follows the step (not the last report)
    ft2 = run(fw, sched('E1,enc_loss,FR,3010,3020,\n'))['fault_tick']
    assert ft2[np.isclose(ft2[:, 0], 3.01), 1] == 2 and ft2[np.isclose(ft2[:, 0], 3.02), 1] == 0
    assert log['faults']['events'] == [[3000, 'E1', 1, 'on'], [4000, 'E1', 0, 'off']]


def test_imu_loss_drops_reports_then_recovers(fw):
    log = run(fw, sched('L1,imu_loss,imu,3000,4000,\n'))
    t, imu = log['t'], log['imu']
    src, age = imu[:, 2], imu[:, 6]
    assert np.all(src[(t > 2.0) & (t < 3.0)] == 1)
    assert np.all(src[(t > 3.2) & (t < 4.0)] == 0) and age[(t > 3.9) & (t < 4.0)].min() > 800
    assert np.all(src[(t > 4.3)] == 1)


def test_delay_equals_shifted_truth(fw):
    log = run(fw, sched('D1,imu_delay,imu,2000,5000,d_ms=100\n'))
    fi = log['fault_imu']                       # t, mask, true(gz ax ay az yaw), sent(...)
    t, gz_true, gz_sent = fi[:, 0], fi[:, 2], fi[:, 7]
    inside = (t >= 2.0) & (t < 5.0)
    assert np.array_equal(fi[~inside][:, 2:7], fi[~inside][:, 7:12])
    idx = np.where(inside)[0]                   # 100 ms = 2 report periods at 20 Hz
    assert np.array_equal(fi[idx][:, 7:12], fi[idx - 2][:, 2:7])


def test_delay_shifts_the_noise_row_too(fw):
    """A noise whose value differs at every report: the delayed report carries the noise of t - d."""
    class RowNoise:
        def apply(self, s, t_us):
            return (s[0] + 0.01 * ((t_us // 50_000) % 7),) + tuple(s[1:])

    log = run(fw, sched('D1,imu_delay,imu,2000,5000,d_ms=100\n'), imu=dict(rate_hz=20.0, noise=RowNoise()))
    fi = log['fault_imu']
    idx = np.where((fi[:, 0] >= 2.0) & (fi[:, 0] < 5.0))[0]
    assert np.array_equal(fi[idx, 7], fi[idx - 2, 2])


def test_scale_is_the_frame_integer_operation(fw):
    log = run(fw, sched('S1,gyro_scale,gz,2000,5000,s=0.3\n'))
    fi = log['fault_imu']
    inside = (fi[:, 0] >= 2.0) & (fi[:, 0] < 5.0)
    assert inside.sum() == 60
    for gt, gsent in fi[inside][:, [2, 7]]:
        assert _to_int16(gsent, GZR) == fs.scale_raw(_to_int16(gt, GZR), 0.3)
    assert np.array_equal(fi[~inside][:, 2:12], np.hstack([fi[~inside][:, 2:7], fi[~inside][:, 2:7]]))
    assert np.array_equal(fi[inside][:, 3:7], fi[inside][:, 8:12])         # accel, yaw untouched


class _ArgRec:
    """Records the arguments fwsim really passes to the firmware (Codex FI2-7), not the wrapper's own rows."""
    def __init__(self, f):
        self.f, self.reports, self.ticks = f, {}, []

    def __getattr__(self, n):
        return getattr(self.f, n)

    def imu_report(self, kind, t_us, v):
        self.reports[(int(t_us), kind)] = v
        return self.f.imu_report(kind, t_us, v)

    def tick(self, t_us, counts, sp):
        self.ticks.append((int(t_us), [int(c) for c in counts]))
        return self.f.tick(t_us, counts, sp)


def test_pi_and_fwsim_give_the_same_integers(fw):
    """Design 9.4, Codex FI1-11, FI2-7: the true IMU values of a real fault_run (what fwsim computed before
    the fault) and the real plant state of every 1 ms step (wheels turning) go through the Pi path
    (encode_state + FramePatcher, one frame per step). (1) IMU: firmware_view of the int16 the Pi sends
    equals what fw.imu_report() really got, for gyro_scale and imu_delay. (2) Wheels: the Pi frame has
    omega 0 for RL exactly inside the window, the true omega (not 0) for RL outside and for the other wheels
    always; fw.tick() really got counts that stop for RL inside the window and follow the true ones before."""
    from hil_node import plant_to_frame_state
    from imu_model import firmware_view
    from protocol import IMU_RANGES, encode_state
    s = sched('S1,gyro_scale,gz,1000,2500,s=0.3\nD1,imu_delay,imu,2000,3000,d_ms=100\n'
              'E1,enc_loss,RL,1500,2500,\n')
    rec = _ArgRec(fw)
    ff = fault_run.FwsimFaults(s)
    log = ff.run(rec, _sp, 3.5, MecanumParams(), imu=dict(IMU), mode='PID')
    fi = log['fault_imu']
    rep = {int(round(t * 1000)): row for t, row in zip(fi[:, 0], fi)}
    p = fs.FramePatcher(s)
    p.start(0)
    n_checked = enc_in = 0
    for k in range(3500):
        row = rep.get(k)
        state = plant_to_frame_state(list(ff.X[k]))             # state x_k of step k, wheels turning
        imu = tuple(row[2:7]) if row is not None else (0.0,) * 5
        true = encode_state(state, imu=imu, imu_seq=k % 255 + 1)
        sent = p.patch(k, true)
        w_true = [int.from_bytes(true[2 * i:2 * i + 2], 'big', signed=True) for i in range(4)]
        w_sent = [int.from_bytes(sent[2 * i:2 * i + 2], 'big', signed=True) for i in range(4)]
        if 1500 <= k < 2500:
            assert w_sent == w_true[:2] + [0] + w_true[3:], k
            enc_in += w_true[2] != 0
        else:
            assert w_sent == w_true, k
        if row is None:
            continue
        pi = [int.from_bytes(sent[o:o + 2], 'big', signed=True) for o in (22, 31, 33, 35, 37)]
        gz, acc, yaw_deg = firmware_view(tuple(v / 32767.0 * r for v, r in zip(pi, IMU_RANGES)))
        assert rec.reports[(k * 1000, 'gyro')] == gz, k
        assert tuple(rec.reports[(k * 1000, 'accel')]) == acc, k
        assert rec.reports[(k * 1000, 'rv')] == yaw_deg, k
        n_checked += 1
    assert n_checked == 69                                  # 20 Hz reports, 50 ms .. 3450 ms
    assert enc_in > 500                                     # RL really turns inside the window (int16 != 0)
    ft = log['fault_tick']                                  # t, mask, true x4, sent x4 (wheel RL = 2)
    assert [c for _, c in rec.ticks] == ft[:, 6:10].astype(int).tolist()   # what fw.tick() really got
    d_true, d_sent = np.diff(ft[:, 4]), np.diff(ft[:, 8])
    t1 = ft[1:, 0]
    inside = (t1 > 1.5) & (t1 <= 2.5)                       # tick interval (t - 10 ms, t] inside [1500, 2500)
    assert np.all(d_sent[inside] == 0) and np.any(d_true[inside] != 0)
    before = t1 <= 1.5
    assert np.array_equal(d_sent[before], d_true[before])
    # after the window the sent angle is the true one minus a fixed angle: with floor(), the count gap
    # stays within one count of the gap at the end of the window
    gap = ft[1:, 4] - ft[1:, 8]
    g0 = gap[np.nonzero(inside)[0][-1]]
    after = t1 > 2.5
    assert np.all((gap[after] >= g0 - 1) & (gap[after] <= g0 + 1)) and g0 > 0
    assert np.array_equal(ft[:, 2:4], ft[:, 6:8]) and np.array_equal(ft[:, 5], ft[:, 9])   # other wheels


def test_noise_off_paths_are_simulate_paths(fw):
    """Codex FI1-5, FI2-4: with the IMU noise off, an empty schedule and an encoder/IMU-loss schedule leave
    the IMU path of simulate as it is (no firmware_view): the empty one is bit identical, the enc_loss one has
    the same IMU columns up to its window; a delay / scale row with the noise off is refused; with the noise
    on, a delay / scale run equals its baseline up to the window."""
    imu = dict(rate_hz=20.0)
    a = fwsim.simulate(fw, _sp, 4.0, MecanumParams(), imu=dict(imu), mode='PID')
    b = run(fw, fs.Schedule(()), duration=4.0, imu=imu)
    for k in a:
        assert np.array_equal(a[k], b[k], equal_nan=True), k
    assert b['fault_imu'].shape == (0, 12)
    c = run(fw, sched('E1,enc_loss,FL,2000,3000,\n', run_ms=4000), duration=4.0, imu=imu)
    before = a['t'] < 2.0
    assert np.array_equal(a['imu'][before], c['imu'][before])
    # Codex FI2-4: delay / scale with the noise off would change the IMU path of the whole run: refused
    for body in ('S1,gyro_scale,gz,2000,3000,s=0.3\n', 'D1,imu_delay,imu,2000,3000,d_ms=100\n'):
        with pytest.raises(ValueError, match='noise on'):
            run(fw, sched(body, run_ms=4000), duration=4.0, imu=imu)
        assert fwsim.plant_step is fault_run._REAL_PLANT_STEP
    # with the noise on, a scale / delay run and its baseline share the IMU path up to the window
    base = run(fw, fs.Schedule(()), duration=4.0)
    for body in ('S1,gyro_scale,gz,2000,3000,s=0.3\n', 'D1,imu_delay,imu,2000,3000,d_ms=100\n'):
        e = run(fw, sched(body, run_ms=4000), duration=4.0)
        before = base['t'] < 2.0
        for key in ('imu', 'v_fw', 'count', 'duty'):
            assert np.array_equal(base[key][before], e[key][before], equal_nan=True), (body, key)
        fb, fe = base['fault_imu'], e['fault_imu']
        rb = fb[:, 0] < 2.0
        assert np.array_equal(fb[rb], fe[fe[:, 0] < 2.0])


def test_c7_table_reads_real_fault_run_output(fw, tmp_path):
    """tools/fault_c7_table.py on folders written as fault_run.py writes them (fwsim.write_run + write_faults):
    the log columns and meta.yaml lines it assumes are there; an encoder loss on FL shows in the row."""
    sys.path.insert(0, os.path.join(HERE, '..', '..', 'tools'))
    import fault_c7_table as ct
    cfg = dict(task='A-W1', mode='PID', seed=0, imu_rate=20.0, imu_boot_reset=-1.0, imu_noise='replay',
               imu_noise_seed=0, ground=None, mu=None, no_imu=False, duration=6.0)
    for name, s in (('PID_base', fs.Schedule(())), ('PID_encFL', sched('E1,enc_loss,FL,3000,5000,\n'))):
        log = run(fw, s, duration=6.0)
        out = str(tmp_path / name)
        fwsim.write_run(out, log, fwsim.metrics(log), fw, dict(cfg), MecanumParams())
        fault_run.write_faults(out, log)
    (r,) = ct.table(str(tmp_path))
    assert (r['controller'], r['case'], r['win_start_s'], r['win_end_s'], r['complete']) == ('PID', 'encFL', 3.0,
                                                                                         5.0, True)
    assert r['ticks'] == r['ticks_expected'] == 600 and r['imu_rows'] == r['base_imu_rows'] == 120
    assert r['imu_dropped'] == 0 and r['err_rms_vx'] > 5 * r['base_err_rms_vx']


def test_cli_baseline_and_fault_run(tmp_path):
    import subprocess
    p = tmp_path / 's.csv'
    p.write_text(HEAD + 'L1,imu_loss,imu,3000,5000,\n')
    for name, extra in (('base', ['--baseline']), ('loss', ['--faults', str(p)])):
        r = subprocess.run([sys.executable, os.path.join(HERE, '..', 'fault_run.py'), '--task', 'A-W1',
                            '--mode', 'PID', '--out', str(tmp_path / name)] + extra,
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr[-800:]
        files = sorted(os.listdir(tmp_path / name))
        assert files == ['faults.json', 'faults_imu.csv', 'faults_tick.csv', 'log.csv', 'meta.yaml',
                         'metrics.json']
    meta = (tmp_path / 'loss' / 'meta.yaml').read_text(encoding='utf-8')
    assert 'faults: {' in meta and '"complete": true' in meta
    r = subprocess.run([sys.executable, os.path.join(HERE, '..', 'fault_run.py'), '--task', 'A-W1', '--mode',
                        'PID', '--faults', str(p), '--out', str(tmp_path / 'late'), '--imu-rate', '20'],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0
    p.write_text(HEAD + 'L1,imu_loss,imu,3000,9500,\n')                 # A-W1 is 8 s + 1 s tail
    r = subprocess.run([sys.executable, os.path.join(HERE, '..', 'fault_run.py'), '--task', 'A-W1', '--mode',
                        'PID', '--faults', str(p), '--out', str(tmp_path / 'bad')],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode != 0 and 'after the run' in r.stderr
