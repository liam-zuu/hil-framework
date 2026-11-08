"""--finite-run wired into the real 1 kHz loop (simulate_hw, FakeH7, no hardware).
Run phases, setpoints on the wire, plant hold, stop paths, run owner, publisher gate."""
import multiprocessing as mp
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from fake_h7 import FakeH7
from hil_node import HILConfig, HILNode
from plant_interface import MockPlant
from protocol import (CMD_SWA_ON, OFS_FAULT, OFS_STATUS, OFS_TRAJ_FLAGS, STATUS_SWA, TRAJ_NEW, TRAJ_VALID,
                      decode_traj_ref)
from run_hil import _finite_run_config, _owner_tick, resolve_swa
from run_lifecycle import ARM, END, RUN, STOPPING, ControlBox, LifecycleConfig
from trajectory_pub import TrajectoryPublisher, TrajectorySpec

BOTH = TRAJ_VALID | TRAJ_NEW


class CountPlant(MockPlant):
    def __init__(self):
        super().__init__()
        self.steps_done = 0

    def step(self, torques):
        self.steps_done += 1
        return super().step(torques)


class Owner(threading.Thread):
    """Plays the owner of run_hil.py: heartbeat, and `event(phase, box)` once per beat."""

    def __init__(self, box, event=None, beat=True):
        super().__init__(daemon=True)
        self.box, self.event, self.do_beat = box, event, beat
        self.quit = False

    def run(self):
        while not self.quit:
            if self.do_beat:
                self.box.beat(time.monotonic_ns())
            if self.event:
                self.event(self.box.phase(), self.box)
            time.sleep(0.002)


def run_node(tmp_path=None, faults=(), event=None, beat=True, ref=(0.4, 0.0, 0.1), box=None,
             swa_mode='off', fake_out=None, **lc):
    """One finite run; returns the node, what went out in each phase, the plant, the box.
    fake_out: list that receives the FakeH7 (I8 tests read its SWA state)."""
    cfg = dict(run_s=0.15, arm_h7_fresh_s=0.12, stop_s=0.06, arm_timeout_s=0.5, heartbeat_max_s=0.1,
               h7_gap_max_s=0.05,   # time.monotonic ticks every 15.6 ms on Windows
               run_grace_s=0.1)
    cfg.update(lc)
    box = box or ControlBox()
    box.beat(time.monotonic_ns())
    fake = FakeH7(faults=faults)
    if fake_out is not None:
        fake_out.append(fake)
    plant = CountPlant()
    sent = []
    shared_ref = mp.Array('d', 4)
    shared_ref[:] = [ref[0], ref[1], ref[2], 1.0]
    flog = str(tmp_path / 'f.npz') if tmp_path else ''
    node = HILNode(plant, HILConfig(simulate_hw=True, max_steps=3000, lifecycle=LifecycleConfig(**cfg),
                                    frame_log=flog, frame_log_max=3000, swa_mode=swa_mode),
                   shared_ref=shared_ref, control=box,
                   responder=lambda m: (sent.append((node._lc.phase, plant.steps_done, bytes(m))), fake(m))[1])
    owner = Owner(box, event, beat)
    owner.start()
    node.run()
    owner.quit = True
    owner.join()
    return node, sent, plant, box


def done_after(seconds):
    t0 = []

    def ev(phase, box):
        if phase == RUN:
            t0.append(time.monotonic())
            if t0[-1] - t0[0] >= seconds:
                box.publisher_done()
    return ev


def zeros_every_20(frames):
    idx = [i for i, (_, _, m) in enumerate(frames) if m[OFS_TRAJ_FLAGS] == BOTH]
    assert idx and idx[0] == 0
    assert all(b - a == 20 for a, b in zip(idx, idx[1:]))
    assert all(decode_traj_ref(frames[i][2])[:3] == (0.0, 0.0, 0.0) for i in idx)
    assert all(m[OFS_TRAJ_FLAGS] == 0 for i, (_, _, m) in enumerate(frames) if i not in idx)


def phases(node):
    return [t['phase'] for t in node.lifecycle_summary()['transitions']]


def test_completed_run_end_to_end(caplog, tmp_path):
    caplog.set_level('INFO')
    node, sent, plant, box = run_node(tmp_path, event=done_after(0.08))
    s = node.lifecycle_summary()
    assert s['outcome'] == 'completed' and not node.run_failed
    assert phases(node) == [ARM, RUN, STOPPING, END]
    assert box.phase() == END

    arm = [f for f in sent if f[0] == ARM]
    run = [f for f in sent if f[0] == RUN]
    stop = [f for f in sent if f[0] == STOPPING]
    assert arm and run and stop
    # ARM: zero with VALID|NEW every 20 steps, the publisher does not get through, plant held
    zeros_every_20(arm[:-1])
    assert all(p == 0 for _, p, _ in arm)
    # RUN: the setpoint of the publisher, flagged once (the count does not change here)
    assert sum(m[OFS_TRAJ_FLAGS] == BOTH for _, _, m in run) == 1
    assert decode_traj_ref(next(m for _, _, m in run if m[OFS_TRAJ_FLAGS]))[0] == pytest.approx(0.4, abs=1e-3)
    # STOPPING: zero from the loop itself on its first step, plant still runs
    zeros_every_20(stop)
    assert stop[-1][1] > stop[0][1]
    assert s['first_zero_step'] is not None

    d = np.load(tmp_path / 'f.npz')
    import json
    m = json.loads(str(d['manifest']))
    assert m['stop_reason'] == 'completed' and m['lifecycle']['outcome'] == 'completed'
    assert 'reason=completed' in caplog.text and '#run_summary,' in caplog.text


def test_plant_integrates_from_the_first_run_step():
    node, sent, plant, _ = run_node(event=done_after(0.03))
    first_run = next(i for i, f in enumerate(sent) if f[0] == RUN)
    assert sent[first_run][1] == 0 and plant.steps_done == len(sent) - first_run


def test_stop_in_run_is_user_stop_and_sends_zero():
    def ev(phase, box):
        if phase == RUN:
            box.request_stop()
    node, sent, _, _ = run_node(event=ev)
    assert node.lifecycle_summary()['outcome'] == 'user_stop' and not node.run_failed
    zeros_every_20([f for f in sent if f[0] == STOPPING])


def test_stop_in_arm_never_reaches_run():
    node, sent, _, _ = run_node(event=lambda phase, box: box.request_stop(), arm_h7_fresh_s=0.2)
    assert RUN not in phases(node) and node.lifecycle_summary()['outcome'] == 'user_stop'


def test_owner_gone_is_error_heartbeat_with_zeros():
    node, sent, _, _ = run_node(beat=False)
    assert node.lifecycle_summary()['outcome'] == 'error:heartbeat' and node.run_failed
    zeros_every_20([f for f in sent if f[0] == STOPPING])


def test_publisher_dead_wins_over_a_later_done():
    def ev(phase, box):
        if phase == RUN:
            box.publisher_dead()
            box.publisher_done()
    node, _, _, _ = run_node(event=ev)
    assert node.lifecycle_summary()['outcome'] == 'error:publisher' and node.run_failed


def test_silent_h7_times_out_in_arm():
    node, sent, plant, _ = run_node(faults=[(0, 10**9, 'silent')], arm_timeout_s=0.3, heartbeat_max_s=5.0)
    s = node.lifecycle_summary()
    assert s['outcome'] == 'error:arm_timeout' and s['detail'] == 'no_h7' and RUN not in phases(node)
    assert not any(f[0] == RUN for f in sent)


def test_run_without_done_times_out():
    node, _, _, _ = run_node(run_s=0.05, run_grace_s=0.05)
    assert node.lifecycle_summary()['outcome'] == 'error:run_timeout'


class HeldLock:
    """The lock as another process holds it: a try fails, a blocking caller gets through."""

    def acquire(self, block=True):
        assert block is False, 'the 1 kHz loop must never wait for the lock'
        return False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_rt_poll_miss_keeps_the_last_values():
    box = ControlBox()
    box.beat(123)
    assert box.rt_poll(RUN) and box.heartbeat_ns == 123
    box.beat(456)
    box._a.get_lock = lambda: HeldLock()
    assert not box.rt_poll(STOPPING)
    assert box.heartbeat_ns == 123 and box.lock_misses == 1 and box.phase() == RUN


def test_lock_never_free_ends_in_error_heartbeat():
    box = ControlBox()
    box.beat(time.monotonic_ns())
    box._a.get_lock = lambda: HeldLock()
    node, sent, _, _ = run_node(box=box, beat=False)
    s = node.lifecycle_summary()
    assert s['outcome'] == 'error:heartbeat' and s['lock_misses'] == len(sent)
    zeros_every_20([f for f in sent if f[0] == STOPPING])


def test_no_lifecycle_needs_no_box_and_keeps_the_old_loop():
    node = HILNode(MockPlant(), HILConfig(simulate_hw=True, max_steps=20))
    node.run()
    assert node.lifecycle_summary() is None and not node.run_failed and node._stop_reason == 'max_steps'


def test_lifecycle_without_box_is_refused():
    with pytest.raises(ValueError):
        HILNode(MockPlant(), HILConfig(simulate_hw=True, lifecycle=LifecycleConfig(run_s=1.0)))


# ── run owner ──────────────────────────────────────────────────────────

@pytest.mark.parametrize('alive, code, stopped, done, dead', [
    (True, None, False, 0, 0),
    (False, 0, False, 1, 0),
    (False, 1, False, 0, 1),
    (False, -15, True, 0, 0),         # Ctrl+C reached the publisher too: still user_stop
    (False, 0, True, 0, 0),
])
def test_owner_tick(alive, code, stopped, done, dead):
    box = ControlBox()
    if stopped:
        box.request_stop()
    _owner_tick(box, SimpleNamespace(is_alive=lambda: alive, exitcode=code))
    assert box.rt_poll(ARM)
    assert (box.pub_done, box.pub_dead) == (bool(done), bool(dead)) and box.heartbeat_ns > 0


def _args(**kw):
    a = dict(steps=0, no_traj=False, traj_loop=False, fixed_omega=None, omega_steps=None,
             omega_schedule=None, load_schedule=None)
    a.update(kw)
    return SimpleNamespace(**a)


@pytest.mark.parametrize('kw', [dict(steps=100), dict(no_traj=True), dict(traj_loop=True),
                                dict(fixed_omega='1,1,1,1'), dict(omega_steps='1,1,1,1:1'),
                                dict(omega_schedule='x.csv'), dict(load_schedule='x.csv')])
def test_finite_run_refuses_other_ends(kw):
    with pytest.raises(SystemExit):
        _finite_run_config(_args(**kw), TrajectorySpec('line', duration=3.0))


def test_finite_run_needs_a_finite_trajectory():
    with pytest.raises(SystemExit):
        _finite_run_config(_args(), TrajectorySpec('line', duration=0.0))


def test_ceiling_is_above_every_phase_limit():
    lc, ceiling = _finite_run_config(_args(), TrajectorySpec('line', duration=3.0))
    assert lc.run_s == 3.0
    assert ceiling / 1000 > lc.arm_timeout_s + lc.run_s + lc.run_grace_s + lc.stop_s


# ── publisher ──────────────────────────────────────────────────────────

def _pub(box, duration=0.06):
    ref = mp.Array('d', 4)
    pub = TrajectoryPublisher(spec=TrajectorySpec('line', v=0.3, duration=duration), simulate=True,
                              shared_ref=ref, control=box)
    return pub, ref


def _run_pub(pub):
    th = threading.Thread(target=_no_signal_run, args=(pub,), daemon=True)
    th.start()
    return th


def _no_signal_run(pub):
    import trajectory_pub as tp
    real, tp.signal.signal = tp.signal.signal, lambda *a: None
    try:
        pub.run()
    finally:
        tp.signal.signal = real


def test_publisher_waits_for_run_and_its_time_starts_there():
    box = ControlBox()
    box.set_phase(ARM)
    pub, ref = _pub(box)
    th = _run_pub(pub)
    time.sleep(0.15)                       # longer than the whole trajectory
    assert th.is_alive() and ref[3] == 0.0
    box.set_phase(RUN)
    th.join(1.0)
    assert not th.is_alive() and ref[3] >= 2


def test_publisher_never_starts_when_the_run_ends_in_arm():
    box = ControlBox()
    box.set_phase(STOPPING)
    pub, ref = _pub(box)
    th = _run_pub(pub)
    th.join(1.0)
    assert not th.is_alive() and ref[3] == 0.0


def test_publisher_without_box_starts_at_once():
    pub, ref = _pub(None)
    th = _run_pub(pub)
    th.join(1.0)
    assert not th.is_alive() and ref[3] >= 2


def test_ctrl_c_does_not_stop_the_loop_directly():
    import signal
    seen = []

    def ev(phase, box):
        if phase == RUN and not seen:
            seen.append(signal.getsignal(signal.SIGINT))
            box.request_stop()
    run_node(event=ev)
    assert seen == [signal.SIG_IGN]


# ── I8: SWA bit (MOSI byte 20 bit 3), Liam 09/11 "Duyet ca 5" ──

def _swa(m):
    return 1 if m[OFS_FAULT] & CMD_SWA_ON else 0


def test_swa_phase_on_only_in_run_and_stopping(tmp_path):
    fk = []
    node, sent, _, _ = run_node(tmp_path, event=done_after(0.08), swa_mode='phase', fake_out=fk)
    assert phases(node) == [ARM, RUN, STOPPING, END]
    for ph, _, m in sent:
        assert _swa(m) == (1 if ph in (RUN, STOPPING) else 0), ph
        assert m[OFS_FAULT] & ~CMD_SWA_ON == 0              # no fault bit set by the SWA path
    assert any(ph == END for ph, _, _ in sent) and _swa(sent[-1][2]) == 0   # last frame says off
    ev = node.swa_events()
    assert [v for _, v in ev] == [0, 1, 0]                  # off in ARM, on at RUN, off at END
    run_first = next(i for i, (ph, _, _) in enumerate(sent) if ph == RUN)
    assert ev[1][0] == run_first
    import json
    m = json.loads(str(np.load(tmp_path / 'f.npz')['manifest']))
    assert m['swa_mode'] == 'phase' and m['swa_events'] == ev
    assert fk[0].swa == 0 and fk[0].swa_changes == 2        # the fake H7 followed the bit


def test_swa_status_bit_echoes_what_the_h7_took():
    node, sent, _, _ = run_node(event=done_after(0.08), swa_mode='phase')
    got = []
    resp = []
    fake = FakeH7()
    for ph, _, m in sent:
        resp.append(fake(m))
        got.append(_swa(m))
    # MISO of frame k carries the SWA the fake H7 took from frame k-1 (one frame late, like the echo)
    st = [r[OFS_STATUS] & STATUS_SWA for r in resp]
    assert st[1:] == got[:-1]


def test_swa_on_and_off_are_fixed():
    for mode, want in (('on', 1), ('off', 0)):
        _, sent, _, _ = run_node(event=done_after(0.05), swa_mode=mode)
        assert {_swa(m) for _, _, m in sent} == {want}


def test_swa_default_is_off_without_lifecycle():
    node = HILNode(MockPlant(), HILConfig(simulate_hw=True, max_steps=20))
    node.run()
    assert node.cfg.swa_mode == 'off' and node.swa_events() == [[0, 0]]


def test_swa_phase_without_lifecycle_refused():
    with pytest.raises(ValueError):
        HILNode(MockPlant(), HILConfig(simulate_hw=True, swa_mode='phase'))
    with pytest.raises(ValueError):
        HILNode(MockPlant(), HILConfig(simulate_hw=True, swa_mode='yes'))


@pytest.mark.parametrize('swa, finite, n_modes, traj, want', [
    ('auto', True, 0, 'line', 'phase'),
    ('auto', False, 0, 'circle', 'on'),        # old plant run with a trajectory: as before
    ('auto', False, 0, 'idle', 'off'),         # --imu-static / --boot: diagnostics
    ('auto', False, 1, 'line', 'off'),         # an omega mode
    ('on', True, 0, 'line', 'on'),
    ('off', False, 0, 'circle', 'off'),
])
def test_resolve_swa(swa, finite, n_modes, traj, want):
    assert resolve_swa(swa, finite, n_modes, traj) == want
