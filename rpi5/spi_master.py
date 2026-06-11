"""
SPI Master — RPi5 → STM32 Nucleo H7

Thin wrapper around spidev for the HIL SPI bus.
Every transfer is full-duplex: MOSI carries state → H7,
MISO carries torques ← H7, both simultaneously.

Hardware config:
  Interface:  /dev/spidev0.0  (SPI bus 0, CE0)
  Mode:       SPI Mode 0 (CPOL=0, CPHA=0) — matches H7 SPI slave default
  Speed:      1 MHz initial  →  increase to 4 MHz after loopback validation
  Frame:      24 bytes (see protocol.py for layout)
  CS:         Hardware (GPIO8 / CE0), active low

Enable SPI on RPi5 (if not already):
  sudo raspi-config → Interface Options → SPI → Enable
  reboot, then check: ls /dev/spidev*

Install spidev:
  pip install spidev

Wiring (SPI):
  RPi5 GPIO8  (CE0,  pin 24) ──→ H7 NSS   (active low)
  RPi5 GPIO11 (SCLK, pin 23) ──→ H7 SCK
  RPi5 GPIO10 (MOSI, pin 19) ──→ H7 MOSI
  RPi5 GPIO9  (MISO, pin 21) ←── H7 MISO
  RPi5 GND    (pin   6/9/14) ──→ H7 GND   (common ground)
"""
from __future__ import annotations

from typing import Optional

try:
    import spidev as _spidev
    _SPIDEV_OK = True
except ImportError:
    _spidev    = None
    _SPIDEV_OK = False

from protocol import FRAME_SIZE


class SPIMaster:
    """
    Full-duplex SPI master (RPi5 side).

    In simulate=True mode, MISO = MOSI echo (software loopback).
    This lets you test protocol framing without real hardware attached.
    """

    def __init__(
        self,
        bus:       int   = 0,
        device:    int   = 0,
        speed_hz:  int   = 1_000_000,
        simulate:  bool  = False,
    ):
        """
        Args:
            bus:      SPI bus index (0 → /dev/spidev0.x).
            device:   SPI device index (0 → /dev/spidev0.0, CE0).
            speed_hz: SPI clock [Hz]. Start at 1 MHz; increase after HW validation.
            simulate: Loopback mode — no real hardware needed.
        """
        self.bus       = bus
        self.device    = device
        self.speed_hz  = speed_hz
        self.simulate  = simulate

        self._spi: Optional[object] = None
        self._n_transfers: int = 0
        self._n_errors:    int = 0

        if not simulate:
            self._open()

    # ── Hardware init ─────────────────────────────────────────────────────

    def _open(self) -> None:
        if not _SPIDEV_OK:
            raise ImportError(
                "spidev is not installed.  Run: pip install spidev\n"
                "Also enable SPI: sudo raspi-config → Interface → SPI"
            )
        spi = _spidev.SpiDev()
        spi.open(self.bus, self.device)
        spi.max_speed_hz  = self.speed_hz
        spi.mode          = 0b00      # Mode 0: CPOL=0, CPHA=0
        spi.bits_per_word = 8
        spi.no_cs         = False     # hardware CE0 (GPIO8)
        spi.lsbfirst      = False     # MSB first — matches H7 SPI default
        self._spi = spi
        print(
            f"[spi] /dev/spidev{self.bus}.{self.device}  "
            f"{self.speed_hz // 1_000} kHz  Mode 0  MSB-first",
            flush=True,
        )

    # ── Core method ───────────────────────────────────────────────────────

    def transfer(self, tx: bytes) -> bytes:
        """
        Perform one full-duplex SPI transaction.

        Args:
            tx: MOSI payload, exactly FRAME_SIZE (24) bytes.

        Returns:
            MISO payload, exactly FRAME_SIZE bytes.

        Raises:
            ValueError:   Wrong tx length.
            RuntimeError: SPI driver error (logged, re-raised).
        """
        if len(tx) != FRAME_SIZE:
            raise ValueError(
                f"SPI frame must be {FRAME_SIZE} bytes, got {len(tx)}"
            )

        self._n_transfers += 1

        if self.simulate:
            # Loopback: H7 echoes MOSI back as MISO (for protocol tests)
            return bytes(tx)

        try:
            rx = self._spi.xfer2(list(tx))
            return bytes(rx)
        except Exception as exc:
            self._n_errors += 1
            raise RuntimeError(f"SPI transfer failed: {exc}") from exc

    # ── Speed adjustment ──────────────────────────────────────────────────

    def set_speed(self, speed_hz: int) -> None:
        """
        Change SPI clock speed at runtime.

        Upgrade path:
          1 MHz → loopback test → 4 MHz → full integration test → production
        """
        self.speed_hz = speed_hz
        if self._spi is not None:
            self._spi.max_speed_hz = speed_hz
        print(f"[spi] Speed → {speed_hz // 1_000} kHz", flush=True)

    # ── Cleanup ───────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._spi is not None:
            self._spi.close()
            self._spi = None

    def __enter__(self) -> 'SPIMaster':
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    # ── Diagnostics ───────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            'transfers': self._n_transfers,
            'errors':    self._n_errors,
            'error_pct': (100 * self._n_errors / max(1, self._n_transfers)),
        }
