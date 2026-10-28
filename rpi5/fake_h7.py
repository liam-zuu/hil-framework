"""
Fake H7 for PC tests — answers SPI frames the way the H7 firmware does, so the
Pi side (run phases, STOP, error paths) can be tested without hardware.

Workflow v3.1 section 9.1 (Codex round 2 #32; Liam approved B0 29/10; B0
review round 1 B11-B14). The `--mock` echo of SPIMaster stays for protocol
tests. What this models, from hil-framework ctrl-imu 877e23c (main.c:199-244,
spi1_handler.c:86-152, uart_relay.c:48-68):

  * One TX buffer. Every SPI frame clocks out whatever the buffer holds; the
    SPI callback re-arms the DMA on the same buffer at once. The main loop,
    when it runs, takes the LATEST received frame (frame_ready is a flag, not
    a queue), relays its trajectory reference when traj_flags has VALID and
    NEW, copies the PWMCapture duty into the MISO *torque* field (pwm field
    left 0, status 0, int16 by C truncation) and writes the buffer with that
    frame's seq. So the echo is >= 1 frame late; how late on the board is not
    measured yet (S06 T1).
  * main_every: the main loop handles a frame after every k-th SPI frame
    (k = 1: echo one frame late; k > 1: a slow main loop skips seqs).
  * faults: (start, end, kind) over frame indices [start, end):
      'stall'      main loop does not run (like the blocking debug printf of
                   Codex S06 #14): MISO keeps the last prepared frame;
      'silent'     transaction lost: MISO all silent_byte, H7 never sees MOSI;
      'miso_bad'   H7 receives MOSI but MISO is all silent_byte;
      'torn'       MISO bytes from torn_at on are zero, as when _encode's
                   memset races the DMA (Codex B2 / THESIS S06 #2);
      'xfer_error' the transfer raises (SPIMaster turns it into RuntimeError).

Not modelled (B12): the 50 ms SPI link watchdog (it stops EncoderGen, it does
not force PWM to 0), the USART2 FIFO dropping a relay packet while busy, the
BNO085 emulator traffic. relay_attempted counts VALID|NEW frames the main loop
saw; it is not evidence that a packet reached the DUT.

Usage:
    fake = FakeH7(duty_fn=lambda n, s, t: [0.2] * 4, faults=[(100, 150, 'stall')])
    spi  = SPIMaster(simulate=True, responder=fake)
"""
from __future__ import annotations

import struct
from typing import Callable, Optional, Sequence

from protocol import (
    FRAME_SIZE, OFS_MISO_SEQ, OFS_PWM, OFS_SEQ, OFS_STATUS, TRAJ_NEW, TRAJ_VALID,
    decode_imu, decode_state, decode_traj_ref,
)

FAULT_KINDS = ('stall', 'silent', 'miso_bad', 'torn', 'xfer_error')
INT16_MAXF = 32767.0


class FakeH7TransferError(Exception):
    """Raised by a 'xfer_error' frame."""


def _c_int16(x: float) -> int:
    """(int16_t)(clamp(x, -1, 1) * 32767.0f) as in spi1_handler.c _encode: truncates toward 0."""
    return int(max(-1.0, min(1.0, float(x))) * INT16_MAXF)


def encode_miso_like_h7(duty, seq: int, status: int = 0) -> bytes:
    """MISO frame as the H7 builds it: torque field = duty, pwm field 0 (main.c:217-224)."""
    buf = bytearray(FRAME_SIZE)
    for i, d in enumerate(duty):
        struct.pack_into('>h', buf, i * 2, _c_int16(d))
        struct.pack_into('>h', buf, OFS_PWM + i * 2, 0)
    buf[OFS_STATUS] = status & 0xFF
    buf[OFS_MISO_SEQ] = seq & 0xFF
    return bytes(buf)


class FakeH7:
    def __init__(
        self,
        duty_fn:     Optional[Callable] = None,
        main_every:  int = 1,
        faults:      Sequence[tuple] = (),
        silent_byte: int = 0x00,
        torn_at:     int = 4,
    ):
        if main_every < 1:
            raise ValueError("main_every must be >= 1")
        for f in faults:
            if len(f) != 3 or f[2] not in FAULT_KINDS or f[0] >= f[1]:
                raise ValueError(f"bad fault {f!r}: (start, end, kind in {FAULT_KINDS})")
        if not 0 <= torn_at < FRAME_SIZE:
            raise ValueError("torn_at must be a byte index in the frame")
        self.duty_fn     = duty_fn or (lambda n, state, traj: [0.0] * 4)
        self.main_every  = main_every
        self.faults      = list(faults)
        self.silent_byte = silent_byte & 0xFF
        self.torn_at     = torn_at

        self._tx = bytes(FRAME_SIZE)          # what the DMA clocks out next
        self._latest: Optional[bytes] = None  # last MOSI received (frame_ready)

        # What a test can check
        self.frames          = 0      # SPI frames clocked
        self.processed       = 0      # main loop runs that handled a frame
        self.processed_seqs: list = []
        self.relay_attempted = 0      # VALID|NEW frames the main loop handed to the relay
        self.last_relay: Optional[tuple] = None    # (vx, vy, wz) of the last one
        self.last_state: Optional[list] = None
        self.imu_updates = 0          # handled frames with a new imu_seq
        self._imu_seq: Optional[int] = None

    def _fault(self, n: int) -> Optional[str]:
        for start, end, kind in self.faults:
            if start <= n < end:
                return kind
        return None

    def __call__(self, mosi: bytes) -> bytes:
        if len(mosi) != FRAME_SIZE:
            raise ValueError(f"SPI frame must be {FRAME_SIZE} bytes, got {len(mosi)}")
        n = self.frames
        self.frames += 1
        kind = self._fault(n)

        if kind == 'xfer_error':
            raise FakeH7TransferError(f"fake H7: transfer error at frame {n}")
        if kind == 'silent':
            return bytes([self.silent_byte]) * FRAME_SIZE

        out = self._tx
        if kind == 'miso_bad':
            out = bytes([self.silent_byte]) * FRAME_SIZE
        elif kind == 'torn':
            out = out[:self.torn_at] + bytes(FRAME_SIZE - self.torn_at)
        self._latest = bytes(mosi)

        if kind != 'stall' and (n + 1) % self.main_every == 0:
            self._main_loop(n)
        return out

    def _main_loop(self, n: int) -> None:
        """The H7 main loop on the latest finished frame: relay, read duty, write the TX buffer."""
        mosi, self._latest = self._latest, None
        if mosi is None:
            return
        self.processed += 1
        self.processed_seqs.append(mosi[OFS_SEQ])
        state = decode_state(mosi)
        vx, vy, wz, flags = decode_traj_ref(mosi)
        if (flags & (TRAJ_VALID | TRAJ_NEW)) == (TRAJ_VALID | TRAJ_NEW):
            self.relay_attempted += 1
            self.last_relay = (vx, vy, wz)
        imu_seq = decode_imu(mosi)[-1]
        if imu_seq != self._imu_seq:
            self.imu_updates += self._imu_seq is not None
            self._imu_seq = imu_seq
        self.last_state = state

        duty = list(self.duty_fn(n, state, (vx, vy, wz, flags)))
        if len(duty) != 4:
            raise ValueError("duty_fn must return 4 values")
        self._tx = encode_miso_like_h7(duty, seq=mosi[OFS_SEQ])
