"""
Fake H7 for PC tests — answers SPI frames the way the H7 firmware does, so the
Pi side (run phases, STOP, error paths) can be tested without hardware.

Workflow v3.1 section 9.1 (Codex round 2 #32; Liam approved B0 29/10). The
`--mock` echo of SPIMaster stays for protocol tests; this one returns a MISO
frame with the real layout:

  * What the H7 does (hil-framework ctrl-imu 877e23c, main.c:217-224,
    spi1_handler.c:86-98, 137-151): the main loop picks up a finished frame,
    relays the trajectory reference to the ESP32 when traj_flags has VALID and
    NEW, copies the PWMCapture duty into the MISO *torque* field (pwm field
    left 0, status 0) and echoes that frame's seq. The SPI callback re-arms the
    DMA with the same _tx buffer at once, so the echo shows up one frame later
    or more; how many frames on the board is not measured yet (S06 T1).
  * seq_lag: frames between a MOSI frame and the MISO frame carrying its echo.
  * duty_fn(frame_index, state, traj) -> 4 duties in [-1, 1]; default zeros.
  * faults: (start, end, kind) over frame indices [start, end):
      'stall'  — the main loop does not run (like the blocking debug printf of
                 Codex S06 #14): MISO repeats the last frame, no seq advance,
                 no relay;
      'silent' — the H7 does not answer at all: every MISO byte = silent_byte.

Usage:
    fake = FakeH7(duty_fn=lambda n, s, t: [0.2] * 4, seq_lag=2,
                  faults=[(100, 150, 'stall')])
    spi  = SPIMaster(simulate=True, responder=fake)
"""
from __future__ import annotations

from collections import deque
from typing import Callable, Optional, Sequence

from protocol import (
    FRAME_SIZE, OFS_SEQ, TRAJ_NEW, TRAJ_VALID,
    decode_imu, decode_state, decode_traj_ref, encode_torques,
)

FAULT_KINDS = ('stall', 'silent')


class FakeH7:
    def __init__(
        self,
        duty_fn:     Optional[Callable] = None,
        seq_lag:     int = 1,
        faults:      Sequence[tuple] = (),
        silent_byte: int = 0x00,
    ):
        if seq_lag < 1:
            raise ValueError("seq_lag must be >= 1 (the H7 cannot echo inside the same frame)")
        for f in faults:
            if len(f) != 3 or f[2] not in FAULT_KINDS or f[0] >= f[1]:
                raise ValueError(f"bad fault {f!r}: (start, end, kind in {FAULT_KINDS})")
        self.duty_fn     = duty_fn or (lambda n, state, traj: [0.0] * 4)
        self.seq_lag     = seq_lag
        self.faults      = list(faults)
        self.silent_byte = silent_byte & 0xFF

        # MISO frames prepared by the main loop, oldest first; the head goes out next
        self._pending = deque([bytes(FRAME_SIZE)] * seq_lag)
        self._last_out = bytes(FRAME_SIZE)

        # What a test can check
        self.frames      = 0      # SPI frames clocked
        self.processed   = 0      # frames the main loop handled
        self.relays      = 0      # UART packets the H7 would send to the ESP32
        self.last_relay: Optional[tuple] = None    # (vx, vy, wz) of the last relayed packet
        self.last_state: Optional[list] = None
        self.imu_updates = 0      # frames with a new imu_seq
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

        if kind == 'silent':
            return bytes([self.silent_byte]) * FRAME_SIZE
        if kind == 'stall':
            return self._last_out

        out = self._pending.popleft()
        self._last_out = out
        self._process(n, mosi)
        return out

    def _process(self, n: int, mosi: bytes) -> None:
        """The H7 main loop on a finished frame: relay, read duty, prepare the echo."""
        self.processed += 1
        state = decode_state(mosi)
        vx, vy, wz, flags = decode_traj_ref(mosi)
        if (flags & (TRAJ_VALID | TRAJ_NEW)) == (TRAJ_VALID | TRAJ_NEW):
            self.relays += 1
            self.last_relay = (vx, vy, wz)
        imu_seq = decode_imu(mosi)[-1]
        if imu_seq != self._imu_seq:
            self.imu_updates += self._imu_seq is not None
            self._imu_seq = imu_seq
        self.last_state = state

        duty = [max(-1.0, min(1.0, float(d))) for d in self.duty_fn(n, state, (vx, vy, wz, flags))]
        if len(duty) != 4:
            raise ValueError("duty_fn must return 4 values")
        self._pending.append(encode_torques(duty, pwm=None, status=0, seq=mosi[OFS_SEQ]))
