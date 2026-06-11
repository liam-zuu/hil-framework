"""
GPIO Sync Pulse — RPi5 → H7

Generates a short HIGH pulse on a GPIO pin to tell H7 that a new SPI
frame is imminent.  H7's EXTI fires on the rising edge, enabling its
SPI DMA receive.

CRITICAL RPi5 detail:
  RPi5 uses the RP1 I/O controller chip, NOT BCM2835/BCM2711.
  This means:
    - Library:      lgpio  (NOT RPi.GPIO or gpiozero)
    - GPIO device:  /dev/gpiochip4  (NOT /dev/gpiochip0)
    - Pin numbers:  BCM numbering still works (GPIO17 = BCM 17)

  Install:
    sudo apt install python3-lgpio
    pip install lgpio

Wiring:
    RPi5 GPIO17  (physical pin 11) ─────→ H7 EXTI input pin
    RPi5 GND     (physical pin 9)  ─────→ H7 GND    [MANDATORY — common ground]
"""
from __future__ import annotations

import time
from typing import Optional

try:
    import lgpio as _lgpio
    _LGPIO_OK = True
except ImportError:
    _lgpio     = None
    _LGPIO_OK  = False

# RPi5 GPIO chip (RP1 exposes GPIO on /dev/gpiochip4)
_RPI5_CHIP       = 4
_PULSE_WIDTH_US  = 10    # 10 μs pulse — enough for H7 EXTI to catch reliably


class GPIOSync:
    """
    Sync pulse generator: RPi5 → H7.

    Thread-safe only if pulse() is called from a single thread/process.
    """

    def __init__(
        self,
        pin:       int  = 17,        # BCM GPIO number
        gpiochip:  int  = _RPI5_CHIP,
        simulate:  bool = False,     # True = no real hardware
    ):
        """
        Args:
            pin:      BCM GPIO number of the sync output line.
            gpiochip: lgpio chip index.  4 for RPi5, 0 for RPi4.
            simulate: Skip hardware initialisation (for dev / CI testing).
        """
        self.pin       = pin
        self.gpiochip  = gpiochip
        self.simulate  = simulate

        self._handle: Optional[int] = None
        self._n_pulses: int         = 0

        if not simulate:
            self._open()

    # ── Hardware init ─────────────────────────────────────────────────────

    def _open(self) -> None:
        if not _LGPIO_OK:
            raise ImportError(
                "lgpio is not installed.\n"
                "  sudo apt install python3-lgpio\n"
                "  pip install lgpio"
            )
        self._handle = _lgpio.gpiochip_open(self.gpiochip)
        _lgpio.gpio_claim_output(self._handle, self.pin, 0)   # start LOW
        print(
            f"[gpio] Opened /dev/gpiochip{self.gpiochip}, "
            f"BCM GPIO{self.pin} as sync output",
            flush=True,
        )

    # ── Core method ───────────────────────────────────────────────────────

    def pulse(self) -> None:
        """
        Generate one sync pulse: LOW → HIGH (10 μs) → LOW.

        H7 EXTI fires on the rising edge and prepares its SPI DMA receive.
        Call this before every SPI transfer in the plant loop.

        In simulate mode this is a no-op (timing still advances the counter).
        """
        self._n_pulses += 1
        if self.simulate:
            return

        _lgpio.gpio_write(self._handle, self.pin, 1)
        time.sleep(_PULSE_WIDTH_US * 1e-6)
        _lgpio.gpio_write(self._handle, self.pin, 0)

    # ── Cleanup ───────────────────────────────────────────────────────────

    def close(self) -> None:
        """Drive pin LOW and release the GPIO chip handle."""
        if self._handle is not None and _LGPIO_OK:
            try:
                _lgpio.gpio_write(self._handle, self.pin, 0)
                _lgpio.gpiochip_close(self._handle)
            except Exception:
                pass
            finally:
                self._handle = None

    # ── Context manager ───────────────────────────────────────────────────

    def __enter__(self) -> 'GPIOSync':
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    # ── Diagnostics ───────────────────────────────────────────────────────

    @property
    def pulse_count(self) -> int:
        """Total pulses generated since creation (or last reset)."""
        return self._n_pulses

    def reset_count(self) -> None:
        self._n_pulses = 0
