"""fake_h7.FakeH7 + SPIMaster responder (workflow v3.1 section 9.1, Codex round 2 #32;
Liam approved B0 29/10). No hardware."""
import multiprocessing as mp

import pytest

from fake_h7 import FakeH7
from hil_node import HILConfig, HILNode
from plant_interface import MockPlant
from protocol import (
    FRAME_SIZE, TRAJ_NEW, TRAJ_VALID, decode_output, encode_state,
)
from spi_master import SPIMaster

ZERO_STATE = [0.0] * 10


def mosi(seq, flags=0, ref=(0.0, 0.0, 0.0), imu_seq=0):
    return encode_state(ZERO_STATE, traj_ref=ref, traj_flags=flags, seq=seq,
                        imu=(0.0, 0.0, 0.0, 9.81, 0.0), imu_seq=imu_seq)


@pytest.mark.parametrize('lag', [1, 2, 3])
def test_seq_echo_comes_lag_frames_later(lag):
    fake = FakeH7(seq_lag=lag)
    out = [decode_output(fake(mosi(s))) for s in range(10, 20)]
    assert [o['seq'] for o in out[:lag]] == [0] * lag          # nothing prepared yet
    assert [o['seq'] for o in out[lag:]] == list(range(10, 20 - lag))


def test_duty_in_torque_field_pwm_and_status_zero_like_the_h7():
    fake = FakeH7(duty_fn=lambda n, s, t: [0.1 * (n + 1), -0.5, 1.7, 0.0])
    fake(mosi(1))
    o = decode_output(fake(mosi(2)))
    assert o['torque'] == pytest.approx([0.1, -0.5, 1.0, 0.0], abs=1e-4)   # clamped to 1
    assert o['pwm'] == [0.0] * 4 and o['status'] == 0 and o['seq'] == 1


def test_relay_only_with_valid_and_new():
    fake = FakeH7()
    fake(mosi(1, flags=TRAJ_VALID, ref=(0.3, 0.0, 0.0)))
    fake(mosi(2, flags=TRAJ_NEW, ref=(0.4, 0.0, 0.0)))
    assert fake.relays == 0
    fake(mosi(3, flags=TRAJ_VALID | TRAJ_NEW, ref=(0.5, -0.1, 1.0)))
    assert fake.relays == 1
    assert fake.last_relay == pytest.approx((0.5, -0.1, 1.0), abs=1e-3)


def test_imu_updates_counted_on_seq_change():
    fake = FakeH7()
    for s, iseq in enumerate([0, 0, 1, 1, 2, 3]):
        fake(mosi(s, imu_seq=iseq))
    assert fake.imu_updates == 3


def test_stall_repeats_last_frame_and_skips_processing():
    fake = FakeH7(faults=[(3, 6, 'stall')], duty_fn=lambda n, s, t: [0.0] * 4)
    seqs = []
    for s in range(1, 10):
        seqs.append(decode_output(fake(mosi(s, flags=TRAJ_VALID | TRAJ_NEW)))['seq'])
    # frames 0..2 normal (echo of 0, 1, 2), 3..5 stalled (repeat 2), then the H7 handles frame 6
    assert seqs == [0, 1, 2, 2, 2, 2, 3, 7, 8]
    assert fake.processed == 6 and fake.relays == 6


def test_silent_returns_fill_byte():
    fake = FakeH7(faults=[(1, 3, 'silent')], silent_byte=0xFF)
    fake(mosi(1))
    assert fake(mosi(2)) == b'\xff' * FRAME_SIZE
    assert fake(mosi(3)) == b'\xff' * FRAME_SIZE
    assert decode_output(fake(mosi(4)))['seq'] == 1


def test_bad_arguments():
    with pytest.raises(ValueError):
        FakeH7(seq_lag=0)
    with pytest.raises(ValueError):
        FakeH7(faults=[(5, 5, 'stall')])
    with pytest.raises(ValueError):
        FakeH7(faults=[(0, 5, 'unknown')])
    with pytest.raises(ValueError):
        SPIMaster(simulate=False, responder=FakeH7())


def test_spi_master_uses_responder_and_echo_stays_default():
    frame = mosi(7)
    assert SPIMaster(simulate=True).transfer(frame) == frame
    spi = SPIMaster(simulate=True, responder=FakeH7())
    assert spi.transfer(frame) == bytes(FRAME_SIZE)
    assert decode_output(spi.transfer(mosi(8)))['seq'] == 7
    with pytest.raises(ValueError):
        SPIMaster(simulate=True, responder=lambda tx: b'\x00' * 3).transfer(frame)


class RecPlant(MockPlant):
    def __init__(self):
        super().__init__()
        self.torques = []

    def step(self, torques):
        self.torques.append(list(torques))
        return super().step(torques)


def test_hil_node_reads_duty_from_fake_h7_and_relays_new_setpoints():
    """The Pi loop with the fake H7 in place of the board: plant gets the duty, the H7
    relays only frames where the publisher counter changed."""
    plant = RecPlant()
    ref = mp.Array('d', 4)
    node = HILNode(plant, config=HILConfig(simulate_hw=True, max_steps=0), shared_ref=ref)
    fake = FakeH7(duty_fn=lambda n, s, t: [0.25, -0.25, 0.5, -0.5])
    node._spi = SPIMaster(simulate=True, responder=fake)
    for k in range(40):
        if k % 20 == 0:
            with ref.get_lock():
                ref[:] = [0.3, 0.0, 0.0, ref[3] + 1]
        node._tick(0)
    assert plant.torques[0] == [0.0] * 4                 # first MISO: nothing prepared yet
    assert plant.torques[-1] == pytest.approx([0.25, -0.25, 0.5, -0.5], abs=1e-4)
    assert fake.relays == 2 and fake.last_relay == pytest.approx((0.3, 0.0, 0.0), abs=1e-3)
