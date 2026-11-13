"""Fault injection batch 1 (L82) wired into the 1 kHz loop (simulate_hw, no hardware): the frame log keeps the
true frames, only the windows of the schedule change what is sent, t0 = first RUN step, nothing injected in
ARM / STOPPING / the END tail (design 6.2), manifest and #faults line, run_hil.py --faults checks."""
import json
import multiprocessing as mp
import os
import subprocess
import sys

import numpy as np
import pytest

import fault_schedule as fs
from fake_h7 import FakeH7
from hil_node import HILConfig, HILNode
from mecanum_plant import MecanumPlant
from protocol import CMD_SWA_ON, FRAME_SIZE, OFS_FAULT
from run_lifecycle import ARM, RUN, ControlBox, LifecycleConfig
from tests.test_run_wire import CountPlant, Owner, done_after

HERE = os.path.dirname(os.path.abspath(__file__))
HEAD = fs.MAGIC + '\nid,fault,target,start_ms,end_ms,params\n'
DUTY = bytes([0x10, 0x00] * 4) + bytes(FRAME_SIZE - 8)    # constant duty from the "H7" (rule 3.7: no fixed 40)
IMU_STATIC = (0.5, 0.1, 0.2, 9.81, 0.3)


def schedule(text, run_ms, imu):
    rows = fs.parse(text)
    fs.check(rows, run_ms, imu)
    return fs.Schedule(tuple(rows), 'x.csv', 'f' * 64)


def open_loop(tmp_path, name, faults, steps=400):
    plant = MecanumPlant(dt=0.001, seed=0)
    plant.reset([0, 0, 0, 0.2, 0, 0.8, 10, 12, 14, 16])      # turning from the first step: faults change bytes
    p = str(tmp_path / f'{name}.npz')
    cfg = HILConfig(simulate_hw=True, max_steps=steps, period_us=1_000, frame_log=p, faults=faults)
    node = HILNode(plant, config=cfg)
    node._spi.transfer = lambda mosi: DUTY
    node.run()
    return plant.get_state(), np.load(p), node


def test_without_faults_the_log_has_no_truth_array(tmp_path):
    _, f, node = open_loop(tmp_path, 'a', None)
    assert 'mosi_true' not in f.files and json.loads(str(f['manifest']))['faults'] is None
    assert node.fault_summary() is None


def test_open_loop_windows_only_and_plant_untouched(tmp_path, caplog):
    caplog.set_level('INFO')
    s0, f0, _ = open_loop(tmp_path, 'a', None)
    sch = schedule(HEAD + 'E1,enc_loss,FR,50,120,\nL1,imu_loss,imu,100,200,\n'
                          'D1,imu_delay,imu,220,300,d_ms=20\nS1,gyro_scale,gz,250,350,s=0.3\n', 400, 'plant')
    s1, f1, node = open_loop(tmp_path, 'b', sch)
    assert s0 == s1                                         # the plant ran the same dynamics
    m0, m1, t1 = f0['mosi'], f1['mosi'], f1['mosi_true']
    assert np.array_equal(m0, t1)                           # the truth record = the frame of today's code
    changed = np.nonzero(np.any(m1 != t1, axis=1))[0]
    assert changed.min() >= 50 and changed.max() < 350
    assert np.all(m1[50:120, 2:4] == 0) and np.all(np.any(t1[50:120, 2:4] != 0, axis=1))
    assert np.all(m1[120:, 2:4] == t1[120:, 2:4])
    # every frame of [50, 200) and [220, 300) changes; in [300, 350) only gyro_scale acts: where gz != 0
    gz0 = [k for k in range(300, 350) if not t1[k, 22:24].any()]
    assert set(changed) == set(range(50, 200)) | set(range(220, 350)) - set(gz0)
    assert np.all(m1[100:200, OFS_FAULT] & 4) and not np.any(m1[200:, OFS_FAULT] & 4)
    assert np.array_equal(m1[220:250, 22:24], t1[200:230, 22:24])          # delay only
    assert np.array_equal(m1[220:300, 31:39], t1[200:280, 31:39])          # accel, yaw: delay only
    for k in range(250, 300):                                              # gz: delay, then the scale
        raw = int.from_bytes(bytes(t1[k - 20, 22:24]), 'big', signed=True)
        assert int.from_bytes(bytes(m1[k, 22:24]), 'big', signed=True) == fs.scale_raw(raw, 0.3)
    assert any(t1[k - 20, 22:24].any() for k in range(250, 300))         # the data does exercise it
    man = json.loads(str(f1['manifest']))['faults']
    assert man['t0_step'] == 0 and man['complete'] and man['sha256'] == 'f' * 64
    assert [e[:3] for e in man['events']] == [[50, 'E1', 1], [100, 'L1', 1], [120, 'E1', 0], [200, 'L1', 0],
                                              [220, 'D1', 1], [250, 'S1', 1], [300, 'D1', 0], [350, 'S1', 0]]
    assert any(r.message.startswith('#faults,sha256=ffffffff,rows=4,t0_step=0,events=8,cut=0') for r in caplog.records)


def test_window_open_at_the_last_step_is_cut(tmp_path, caplog):
    caplog.set_level('INFO')
    sch = schedule(HEAD + 'E1,enc_loss,FL,300,500,\n', 500, None)
    _, f, _ = open_loop(tmp_path, 'c', sch, steps=400)        # the loop stops before the window ends
    man = json.loads(str(f['manifest']))['faults']
    assert man['events'][-1] == [400, 'E1', 0, 'cut'] and man['cut'] and not man['complete']
    assert any('lịch không thi hành đủ' in r.message for r in caplog.records)


def test_node_refuses_faults_without_frame_log_or_with_heavy_block(tmp_path):
    sch = schedule(HEAD + 'E1,enc_loss,FL,1,5,\n', 100, None)
    with pytest.raises(ValueError):
        HILNode(MecanumPlant(dt=0.001, seed=0), config=HILConfig(simulate_hw=True, max_steps=10, faults=sch))
    with pytest.raises(ValueError):
        HILNode(MecanumPlant(dt=0.001, seed=0),
                config=HILConfig(simulate_hw=True, max_steps=10, faults=sch, frame_log=str(tmp_path / 'x.npz'),
                                 heavy_block=(1, 5)))


def finite_run(tmp_path, sch, event, **lc):
    """run_node of test_run_wire with a fault schedule, a static IMU, SWA by phase and wheels turning."""
    cfg = dict(run_s=0.15, arm_h7_fresh_s=0.12, stop_s=0.06, arm_timeout_s=0.5, heartbeat_max_s=0.1,
               h7_gap_max_s=0.05, run_grace_s=0.1)
    cfg.update(lc)
    box = ControlBox()
    box.beat(time_ns())
    fake = FakeH7()
    plant = CountPlant()
    plant.reset([5.0, 10.0, 15.0, 20.0] + [0.0] * 6)
    sent = []
    shared_ref = mp.Array('d', 4)
    shared_ref[:] = [0.2, 0.0, 0.0, 1.0]
    flog = str(tmp_path / 'f.npz')
    node = HILNode(plant, HILConfig(simulate_hw=True, max_steps=3000, lifecycle=LifecycleConfig(**cfg),
                                    frame_log=flog, frame_log_max=3000, swa_mode='phase',
                                    imu_static=IMU_STATIC, faults=sch),
                   shared_ref=shared_ref, control=box,
                   responder=lambda m: (sent.append((node._lc.phase, bytes(m))), fake(m))[1])
    owner = Owner(box, event)
    owner.start()
    node.run()
    owner.quit = True
    owner.join()
    return node, sent, np.load(flog)


def time_ns():
    import time
    return time.monotonic_ns()


def test_finite_run_t0_is_first_run_step_and_nothing_outside_run(tmp_path):
    sch = schedule(HEAD + 'E1,enc_loss,FL,10,40,\nL1,imu_loss,imu,20,60,\n', 150, 'static')
    node, sent, f = finite_run(tmp_path, sch, done_after(0.12))
    ph = [p for p, _ in sent]
    k_run = ph.index(RUN)
    man = json.loads(str(f['manifest']))['faults']
    assert man['t0_step'] == k_run and man['complete']
    mosi, true = f['mosi'], f['mosi_true']
    n = len(mosi)
    assert all(bytes(mosi[k]) == m for k, (_, m) in enumerate(sent))      # the log is what went out
    diff = [k for k in range(n) if not np.array_equal(mosi[k], true[k])]
    assert diff and min(diff) == k_run + 10 and max(diff) == k_run + 59
    assert all(ph[k] == RUN for k in diff)
    assert np.all(mosi[k_run + 10:k_run + 40, 0:2] == 0) and np.any(true[k_run + 10:k_run + 40, 0:2] != 0)
    # imu_loss and SWA in one byte: 0x0C in the window (SWA on in RUN), the SWA bit untouched
    assert all(mosi[k, OFS_FAULT] == (CMD_SWA_ON | 4) for k in range(k_run + 20, k_run + 60))
    assert all(mosi[k, OFS_FAULT] & CMD_SWA_ON == true[k, OFS_FAULT] & CMD_SWA_ON for k in range(n))
    assert not any(mosi[k, OFS_FAULT] & 4 for k in range(n) if ph[k] == ARM)


def test_finite_run_ends_inside_a_window_frames_after_are_true(tmp_path):
    """Design 6.2: RUN ends (publisher done) at about 60 ms while the window runs to 140 ms: the window is cut
    at the first step out of RUN, every later frame (STOPPING, END tail) is the true one."""
    sch = schedule(HEAD + 'L1,imu_loss,imu,20,140,\nE1,enc_loss,RR,145,150,\n', 150, 'static')
    node, sent, f = finite_run(tmp_path, sch, done_after(0.06))
    ph = [p for p, _ in sent]
    k_run = ph.index(RUN)
    k_out = next(k for k in range(k_run, len(ph)) if ph[k] != RUN)
    man = json.loads(str(f['manifest']))['faults']
    assert man['cut'] and man['not_run'] == ['E1'] and not man['complete'] and man['stop_step'] == k_out
    assert man['events'][-1] == [k_out - k_run, 'L1', 0, 'cut']
    mosi, true = f['mosi'], f['mosi_true']
    assert all(np.array_equal(mosi[k], true[k]) for k in range(k_out, len(mosi)))
    assert mosi[k_out - 1, OFS_FAULT] & 4


# ── run_hil.py --faults ─────────────────────────────────────────────────────
def _run_hil(tmp_path, body, *extra):
    p = tmp_path / 's.csv'
    p.write_text(HEAD + body)
    return subprocess.run([sys.executable, os.path.join(HERE, '..', 'run_hil.py'), '--mock', '--faults', str(p)]
                          + list(extra), capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize('body,extra,msg', [
    ('E1,enc_loss,FL,1,5,\n', ['--steps', '100'], '--frame-log'),
    ('E1,enc_loss,FL,1,5,\n', ['--frame-log', 'x.npz'], '--finite-run or --steps'),
    ('E1,enc_loss,FL,1,500,\n', ['--steps', '100', '--frame-log', 'x.npz'], 'after the run'),
    ('E1,enc_lost,FL,1,5,\n', ['--steps', '100', '--frame-log', 'x.npz'], 'unknown fault'),
    ('L1,imu_loss,imu,1,5,\n', ['--steps', '100', '--frame-log', 'x.npz'], 'needs an IMU'),
    ('L1,imu_loss,imu,1,5,\n', ['--steps', '100', '--frame-log', 'x.npz', '--fixed-omega', '1,2,3,4'],
     'needs an IMU'),
    ('D1,imu_delay,imu,10,50,d_ms=5\n', ['--steps', '100', '--frame-log', 'x.npz', '--imu-static',
                                         '0.1,0,0,9.81,0'], 'means nothing'),
])
def test_run_hil_refuses(tmp_path, body, extra, msg):
    r = _run_hil(tmp_path, body, *extra)
    assert r.returncode != 0 and msg in r.stderr, r.stderr[-800:]


def test_run_hil_refuses_with_load_schedule(tmp_path):
    import s06_load_schedule as ls
    p = str(tmp_path / 'load.csv')
    ls.write_csv(p, ls.make_schedule(120))
    r = _run_hil(tmp_path, 'E1,enc_loss,FL,1,5,\n', '--load-schedule', p, '--frame-log', 'x.npz')
    assert r.returncode != 0 and '--faults excludes --load-schedule' in r.stderr


def test_run_hil_load_faults_imu_source_and_run_length():
    """load_faults takes the IMU source in the order of HILNode._imu_sample and the RUN length."""
    from types import SimpleNamespace
    from run_hil import load_faults
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 's.csv')
        with open(p, 'w') as f:
            f.write(HEAD + 'S1,gyro_scale,gz,10,1500,s=0.2\n')
        a = SimpleNamespace(frame_log='x.npz', faults=p, steps=0, mock=False, step_log='st.csv')
        cfg = HILConfig(lifecycle=LifecycleConfig(run_s=1.5))
        assert load_faults(a, 0, cfg, False).rows[0].id == 'S1'            # plant IMU, RUN 1500 ms
        cfg = HILConfig(lifecycle=LifecycleConfig(run_s=1.4))
        with pytest.raises(SystemExit):
            load_faults(a, 0, cfg, False)
        a2 = SimpleNamespace(frame_log='x.npz', faults=p, steps=2000, mock=False, step_log='st.csv')
        with pytest.raises(SystemExit):
            load_faults(a2, 1, HILConfig(), False)                         # omega mode: no IMU
        assert load_faults(a2, 1, HILConfig(imu_profile='t4'), False).sha256
        # FI1-10: on the Pi a fault run needs the step log; --mock does not
        with pytest.raises(SystemExit, match='--step-log'):
            load_faults(SimpleNamespace(frame_log='x.npz', faults=p, steps=2000, mock=False, step_log=''),
                        0, HILConfig(), False)
        assert load_faults(SimpleNamespace(frame_log='x.npz', faults=p, steps=2000, mock=True, step_log=''),
                           0, HILConfig(imu_static=IMU_STATIC), False).sha256


@pytest.mark.parametrize('mode', ['--loopback-test', '--jitter-only'])
def test_run_hil_faults_refused_before_the_early_modes(tmp_path, monkeypatch, mode):
    """Codex FI1-3: --faults with a mode that returns before the schedule is read is refused first; the
    loopback test (which opens the SPI) and the jitter test are never called."""
    import run_hil
    called = []
    monkeypatch.setattr(run_hil, '_run_loopback_test', lambda **k: called.append('loopback') or 0)
    monkeypatch.setattr(sys, 'argv', ['run_hil.py', mode, '--faults', str(tmp_path / 'missing.csv')])
    import timing
    monkeypatch.setattr(timing, 'measure_jitter', lambda **k: called.append('jitter'))
    with pytest.raises(SystemExit, match='--faults excludes'):
        run_hil.main()
    assert called == []
