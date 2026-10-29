"""Finite run lifecycle (rpi5/run_lifecycle.py), stage B of the HIL workflow (doc 9.1).

Pure state-machine tests, then one loop through SPIMaster + FakeH7 to check the
zero setpoints of STOPPING reach the H7 relay path and ARM needs H7 answers."""
import pytest

from fake_h7 import FakeH7
from protocol import OFS_MISO_SEQ, TRAJ_NEW, TRAJ_VALID, encode_state
from run_lifecycle import (
    ARM, COMPLETED, END, ERR_ARM_TIMEOUT, ERR_HEARTBEAT, ERR_PREPARE, ERR_PUBLISHER,
    ERR_RUN_TIMEOUT, PREPARE, RUN, STOPPING, USER_STOP, LifecycleConfig, RunLifecycle, StepIn,
)
from spi_master import SPIMaster

DT = 0.001


def drive(lc, n, t0=0.0, **kw):
    """n steps at 1 kHz; kw values may be callables of (i, t). Returns outs and the end time."""
    outs = []
    t = t0
    for i in range(n):
        t = t0 + i * DT
        args = {k: (v(i, t) if callable(v) else v) for k, v in kw.items()}
        outs.append(lc.step(StepIn(t=t, **args)))
        if outs[-1].done:
            break
    return outs, t


def phases(lc):
    return [p for _, _, p in lc.transitions]


def test_happy_path_phases_and_outputs():
    lc = RunLifecycle(LifecycleConfig(run_s=2.0))
    lc.begin(0.0)
    outs, t = drive(lc, 1500, h7_reply_new=True)              # ARM passes after 1 s
    assert lc.phase == RUN
    assert all(o.hold_plant for o in outs if o.phase == ARM)
    assert not any(o.hold_plant for o in outs if o.phase == RUN)
    arm_zero = [i for i, o in enumerate(outs) if o.phase == ARM and o.send_zero]
    assert arm_zero[:3] == [0, 20, 40]                          # 50 Hz zero in ARM
    outs2, t = drive(lc, 5000, t0=t + DT, h7_reply_new=True,
                     publisher_done=lambda i, _t: i >= 1000)
    assert outs2[-1].done and lc.phase == END
    assert lc.outcome == COMPLETED
    assert phases(lc) == [ARM, RUN, STOPPING, END]
    stop = [o for o in outs2 if o.phase == STOPPING]
    assert len(stop) == 1000                                    # stop_s = 1.0 s
    assert [i for i, o in enumerate(stop) if o.send_zero][:3] == [0, 20, 40]
    assert not any(o.publish for o in stop)
    stop_step = lc.transitions[2][0]
    assert lc.first_zero_step == stop_step                      # zero on the first STOPPING step


def test_arm_needs_h7_continuously():
    lc = RunLifecycle(LifecycleConfig(run_s=1.0))
    lc.begin(0.0)
    # answers for 0.9 s, a 50 ms gap, answers again: count restarts after the gap
    drive(lc, 2000, h7_reply_new=lambda i, t: not (0.9 <= t < 0.95))
    run_t = lc.transitions[1][1]
    assert lc.transitions[1][2] == RUN
    assert run_t == pytest.approx(0.95 + 1.0, abs=2 * DT)


def test_arm_timeout_without_h7_goes_through_stopping():
    lc = RunLifecycle(LifecycleConfig(run_s=1.0, arm_timeout_s=0.5))
    lc.begin(0.0)
    outs, _ = drive(lc, 3000, h7_reply_new=False)
    assert outs[-1].done
    assert (lc.outcome, lc.detail) == (ERR_ARM_TIMEOUT, 'no_h7')
    assert phases(lc) == [ARM, STOPPING, END]


def test_need_dut_ready():
    cfg = LifecycleConfig(run_s=1.0, arm_timeout_s=3.0, need_dut_ready=True)
    lc = RunLifecycle(cfg)
    lc.begin(0.0)
    drive(lc, 2000, h7_reply_new=True, dut_ready=lambda i, t: t >= 1.5)
    assert lc.transitions[1][2] == RUN and lc.transitions[1][1] == pytest.approx(1.5, abs=DT)
    assert lc.dut_ready == 'yes'

    lc2 = RunLifecycle(cfg)
    lc2.begin(0.0)
    drive(lc2, 6000, h7_reply_new=True, dut_ready=None)
    assert (lc2.outcome, lc2.detail) == (ERR_ARM_TIMEOUT, 'dut_not_ready')
    assert lc2.dut_ready == 'unverified'


def test_dut_ready_unknown_is_unverified():
    lc = RunLifecycle(LifecycleConfig(run_s=0.5))
    lc.begin(0.0)
    drive(lc, 5000, h7_reply_new=True, publisher_done=lambda i, t: t > 1.2)
    assert lc.outcome == COMPLETED and lc.dut_ready == 'unverified'


@pytest.mark.parametrize('kw, outcome', [
    ({'stop_requested': lambda i, t: t > 1.5}, USER_STOP),
    ({'publisher_alive': lambda i, t: t <= 1.5}, ERR_PUBLISHER),
    ({'heartbeat_age_s': lambda i, t: 3.0 if t > 1.5 else 0.0}, ERR_HEARTBEAT),
])
def test_stop_reasons_in_run(kw, outcome):
    lc = RunLifecycle(LifecycleConfig(run_s=10.0))
    lc.begin(0.0)
    outs, _ = drive(lc, 5000, h7_reply_new=True, **kw)
    assert outs[-1].done and lc.outcome == outcome
    assert phases(lc) == [ARM, RUN, STOPPING, END]


def test_stop_in_arm():
    lc = RunLifecycle(LifecycleConfig(run_s=1.0))
    lc.begin(0.0)
    drive(lc, 3000, h7_reply_new=True, stop_requested=lambda i, t: t > 0.3)
    assert lc.outcome == USER_STOP and phases(lc) == [ARM, STOPPING, END]


def test_first_reason_is_kept():
    lc = RunLifecycle(LifecycleConfig(run_s=10.0))
    lc.begin(0.0)
    # STOP at 1.5 s, publisher dies at 1.6 s (inside STOPPING): outcome stays user_stop
    drive(lc, 5000, h7_reply_new=True, stop_requested=lambda i, t: t > 1.5,
          publisher_alive=lambda i, t: t <= 1.6)
    assert lc.outcome == USER_STOP


def test_same_step_order():
    lc = RunLifecycle(LifecycleConfig(run_s=10.0))
    lc.begin(0.0)
    drive(lc, 1100, h7_reply_new=True)
    assert lc.phase == RUN
    lc.step(StepIn(t=1.2, h7_reply_new=True, publisher_done=True, stop_requested=True,
                   publisher_alive=False, heartbeat_age_s=5.0))
    assert lc.outcome == ERR_HEARTBEAT


def test_run_timeout():
    lc = RunLifecycle(LifecycleConfig(run_s=0.5, run_grace_s=0.2))
    lc.begin(0.0)
    drive(lc, 5000, h7_reply_new=True)                          # publisher never DONE
    run_t = lc.transitions[1][1]
    assert lc.outcome == ERR_RUN_TIMEOUT
    assert lc.transitions[2][1] == pytest.approx(run_t + 0.7, abs=2 * DT)


def test_prepare_failure_skips_stopping():
    lc = RunLifecycle(LifecycleConfig(run_s=1.0))
    lc.fail_prepare(0.0, 'run folder exists')
    assert (lc.phase, lc.outcome, lc.detail) == (END, ERR_PREPARE, 'run folder exists')
    assert phases(lc) == [END]
    with pytest.raises(RuntimeError):
        lc.step(StepIn(t=0.001))


def test_misuse_raises():
    lc = RunLifecycle(LifecycleConfig(run_s=1.0))
    with pytest.raises(RuntimeError):
        lc.step(StepIn(t=0.0))                                  # before begin()
    lc.begin(0.0)
    with pytest.raises(RuntimeError):
        lc.begin(0.1)
    with pytest.raises(ValueError):
        LifecycleConfig(run_s=0.0)
    with pytest.raises(ValueError):
        LifecycleConfig(run_s=1.0, zero_every_steps=0)


def test_summary_fields():
    lc = RunLifecycle(LifecycleConfig(run_s=0.2))
    lc.begin(0.0)
    drive(lc, 5000, h7_reply_new=True, publisher_done=lambda i, t: t > 1.1)
    s = lc.summary()
    assert s['outcome'] == COMPLETED
    assert [x['phase'] for x in s['transitions']] == [ARM, RUN, STOPPING, END]
    assert s['first_zero_step'] == s['transitions'][2]['step']
    assert s['zeros_sent'] == lc.zeros_sent > 0
    assert s['dut_ready'] == 'unverified'


def run_with_fake_h7(fake, run_s=0.5, stop_at=None, n_max=6000):
    """Owner loop on the PC: frames through SPIMaster(simulate) + FakeH7."""
    spi = SPIMaster(simulate=True, responder=fake)
    lc = RunLifecycle(LifecycleConfig(run_s=run_s))
    lc.begin(0.0)
    last_miso_seq = None
    pub_sp = (0.3, 0.0, 0.1)
    out = None
    for i in range(n_max):
        t = i * DT
        flags, ref = 0, (0.0, 0.0, 0.0)
        if out is not None and out.send_zero:
            flags, ref = TRAJ_VALID | TRAJ_NEW, (0.0, 0.0, 0.0)
        elif out is not None and out.publish and i % 20 == 0:
            flags, ref = TRAJ_VALID | TRAJ_NEW, pub_sp
        miso = spi.transfer(encode_state([0.0] * 10, traj_ref=ref, traj_flags=flags, seq=i))
        new = miso[OFS_MISO_SEQ] != last_miso_seq
        last_miso_seq = miso[OFS_MISO_SEQ]
        done = out is not None and out.phase == RUN and t - lc.transitions[1][1] >= run_s
        out = lc.step(StepIn(t=t, h7_reply_new=new, publisher_done=done,
                             stop_requested=stop_at is not None and t >= stop_at))
        if out.done:
            break
    spi.close()
    return lc


def test_with_fake_h7_zero_reaches_relay():
    fake = FakeH7()
    lc = run_with_fake_h7(fake)
    assert lc.outcome == COMPLETED and lc.phase == END
    assert fake.last_relay == (0.0, 0.0, 0.0)                   # last relayed setpoint is the STOPPING zero
    assert fake.relay_attempted > 0


def test_with_fake_h7_stalled_main_loop_delays_arm():
    # H7 main loop stalled for the first 0.5 s: MISO seq does not change, ARM waits
    fake = FakeH7(faults=[(0, 500, 'stall')])
    lc = run_with_fake_h7(fake)
    assert lc.transitions[1][2] == RUN
    assert lc.transitions[1][1] >= 1.5 - 2 * DT


def test_with_fake_h7_silent_link_times_out():
    fake = FakeH7(faults=[(0, 100000, 'silent')])
    spi_lc = run_with_fake_h7(fake, n_max=40000)
    assert (spi_lc.outcome, spi_lc.detail) == (ERR_ARM_TIMEOUT, 'no_h7')
