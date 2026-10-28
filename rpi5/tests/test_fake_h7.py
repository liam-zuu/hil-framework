"""fake_h7.FakeH7 + SPIMaster responder (workflow v3.1 section 9.1, Codex round 2 #32;
Liam approved B0 29/10; B0 review round 1 B11-B14). No hardware.
Expected sequences follow the H7 timeline worked out by hand from main.c / spi1_handler.c
(one TX buffer, main loop takes the latest frame), not from the fake's code."""
import multiprocessing as mp
import struct

import pytest

from fake_h7 import FakeH7, encode_miso_like_h7
from hil_node import HILConfig, HILNode
from plant_interface import MockPlant
from protocol import (
    FRAME_SIZE, TRAJ_NEW, TRAJ_VALID, decode_output, encode_state,
)
from spi_master import SPIMaster

ZERO_STATE = [0.0] * 10
VN = TRAJ_VALID | TRAJ_NEW


def mosi(seq, flags=0, ref=(0.0, 0.0, 0.0), imu_seq=0):
    return encode_state(ZERO_STATE, traj_ref=ref, traj_flags=flags, seq=seq,
                        imu=(0.0, 0.0, 0.0, 9.81, 0.0), imu_seq=imu_seq)


def seqs_out(fake, seqs, **kw):
    return [decode_output(fake(mosi(s, **kw)))['seq'] for s in seqs]


def test_echo_one_frame_late():
    # frame 0 clocks the empty buffer; after each frame the main loop writes that frame's seq
    assert seqs_out(FakeH7(), range(1, 7)) == [0, 1, 2, 3, 4, 5]


def test_slow_main_loop_takes_the_latest_frame_and_skips_seqs():
    # main loop after frames 1, 3, 5 (0-based): handles seq 2, 4, 6
    fake = FakeH7(main_every=2)
    assert seqs_out(fake, range(1, 9)) == [0, 0, 2, 2, 4, 4, 6, 6]
    assert fake.processed_seqs == [2, 4, 6, 8]


def test_stall_keeps_the_last_prepared_frame():
    """Codex B11: after seq 1, 2, 3 the buffer holds seq 3; frames 3-5 stalled clock seq 3;
    the main loop then handles the latest frame (seq 7)."""
    fake = FakeH7(faults=[(3, 6, 'stall')])
    assert seqs_out(fake, range(1, 10), flags=VN) == [0, 1, 2, 3, 3, 3, 3, 7, 8]
    assert fake.processed_seqs == [1, 2, 3, 7, 8, 9] and fake.relay_attempted == 6


def test_duty_as_the_h7_encodes_it():
    """Torque field = duty truncated like (int16_t)(d * 32767.0f); pwm field and status 0."""
    fake = FakeH7(duty_fn=lambda n, s, t: [0.1, -0.5, 1.7, -0.00002])
    fake(mosi(1))
    raw = fake(mosi(2))
    assert struct.unpack_from('>4h', raw, 0) == (3276, -16383, 32767, 0)    # C truncates toward 0
    o = decode_output(raw)
    assert o['pwm'] == [0.0] * 4 and o['status'] == 0 and o['seq'] == 1


@pytest.mark.parametrize('x,want', [
    # Codex B37: the duty is a C float; float32(k/32767) * 32767.0f rounds to exactly k, while
    # the double product (0.99999999907 for k = 1) would truncate to k - 1. Inputs are the
    # float32 values written out; expected values from a float32 product (struct 'f') by hand.
    (3.0518509447574615e-05, 1), (-3.0518509447574615e-05, -1), (6.103701889514923e-05, 2),
    (0.0030518509447574615, 100), (-0.0030518509447574615, -100),
    (0.99999, 32766), (-0.99999, -32766), (1.0, 32767), (-1.0, -32767), (0.0, 0)])
def test_duty_cast_in_float32_next_to_a_step(x, want):
    assert struct.unpack_from('>h', encode_miso_like_h7([x, 0, 0, 0], seq=0), 0)[0] == want


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')])
def test_non_finite_duty_is_refused(bad):
    with pytest.raises(ValueError):
        encode_miso_like_h7([bad, 0, 0, 0], seq=0)


def test_relay_attempted_only_with_valid_and_new():
    fake = FakeH7()
    fake(mosi(1, flags=TRAJ_VALID, ref=(0.3, 0.0, 0.0)))
    fake(mosi(2, flags=TRAJ_NEW, ref=(0.4, 0.0, 0.0)))
    assert fake.relay_attempted == 0
    fake(mosi(3, flags=VN, ref=(0.5, -0.1, 1.0)))
    assert fake.relay_attempted == 1
    assert fake.last_relay == pytest.approx((0.5, -0.1, 1.0), abs=1e-3)


def test_imu_updates_counted_on_seq_change():
    fake = FakeH7()
    for s, iseq in enumerate([0, 0, 1, 1, 2, 3]):
        fake(mosi(s, imu_seq=iseq))
    assert fake.imu_updates == 3


def test_silent_loses_the_transaction_miso_bad_keeps_it():
    fake = FakeH7(faults=[(1, 2, 'silent'), (3, 4, 'miso_bad')], silent_byte=0xFF)
    out = [fake(mosi(s)) for s in range(1, 6)]
    assert out[1] == b'\xff' * FRAME_SIZE and out[3] == b'\xff' * FRAME_SIZE
    assert fake.processed_seqs == [1, 3, 4, 5]          # seq 2 never reached the H7
    assert decode_output(out[4])['seq'] == 4


def test_torn_frame_zeroes_the_tail():
    fake = FakeH7(duty_fn=lambda n, s, t: [0.5] * 4, faults=[(2, 3, 'torn')], torn_at=4)
    fake(mosi(1))
    fake(mosi(2))
    raw = fake(mosi(3))
    assert struct.unpack_from('>4h', raw, 0) == (16383, 16383, 0, 0)
    assert raw[17] == 0                                   # seq byte lost too


def test_transfer_error_goes_through_the_spi_error_path():
    spi = SPIMaster(simulate=True, responder=FakeH7(faults=[(1, 2, 'xfer_error')]))
    spi.transfer(mosi(1))
    with pytest.raises(RuntimeError, match='SPI transfer failed'):
        spi.transfer(mosi(2))
    assert spi.stats['errors'] == 1 and spi.stats['transfers'] == 2
    spi.transfer(mosi(3))                                 # recovers on the next frame


def test_bad_arguments():
    with pytest.raises(ValueError):
        FakeH7(main_every=0)
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
    bad = SPIMaster(simulate=True, responder=lambda tx: b'\x00' * 3)
    with pytest.raises(RuntimeError):
        bad.transfer(frame)
    assert bad.stats['errors'] == 1


def test_encode_matches_the_h7_byte_layout():
    raw = encode_miso_like_h7([1.0, -1.0, 0.0, 0.25], seq=0x1A0 & 0xFF)
    assert raw[:8] == bytes.fromhex('7fff8001 00001fff')
    assert raw[8:16] == bytes(8) and raw[16] == 0 and raw[17] == 0xA0 and raw[18:] == bytes(22)


class RecPlant(MockPlant):
    def __init__(self):
        super().__init__()
        self.torques = []

    def step(self, torques):
        self.torques.append(list(torques))
        return super().step(torques)


def test_hil_node_reads_duty_from_fake_h7_and_relays_new_setpoints():
    """The Pi loop with the fake H7 in place of the board: plant gets the duty, the H7
    sees VALID|NEW only on frames where the publisher counter changed."""
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
    assert fake.relay_attempted == 2 and fake.last_relay == pytest.approx((0.3, 0.0, 0.0), abs=1e-3)
