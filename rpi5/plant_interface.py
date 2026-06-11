"""
Plant Interface — abstract contract for all plant implementations.

Any plant (MecanumPlant, MockPlant, future FMU wrapper…) must implement
PlantInterface so HILNode can swap implementations without code changes.
"""
from __future__ import annotations
from abc import ABC, abstractmethod


class PlantInterface(ABC):
    """Minimal contract that every HIL plant must satisfy."""

    @abstractmethod
    def step(self, torques: list) -> list:
        """
        Advance dynamics by one timestep.

        Called once per HIL tick (every 1 ms).

        Args:
            torques: [τ1, τ2, τ3, τ4] in N·m.
                     Values should already be clipped to ±tau_max.

        Returns:
            New 10-element state vector.
        """

    @abstractmethod
    def get_state(self) -> list:
        """Return current state without advancing time."""

    @abstractmethod
    def reset(self, state: list = None) -> None:
        """
        Reset plant to initial conditions.

        Args:
            state: Optional 10-element initial state.  None → all zeros.
        """


# ── MockPlant ─────────────────────────────────────────────────────────────

class MockPlant(PlantInterface):
    """
    Stationary no-op plant for offline testing.

    State stays at zero; torques are accepted but ignored.
    Useful for:
      - Verifying SPI framing and protocol without running dynamics
      - Timing loop benchmarks without matrix algebra overhead
      - CI pipelines on non-ARM machines
    """

    N_STATES = 10

    def __init__(self):
        self._state: list[float] = [0.0] * self.N_STATES
        self._step_count: int    = 0

    def step(self, torques: list) -> list:
        self._step_count += 1
        return list(self._state)

    def get_state(self) -> list:
        return list(self._state)

    def reset(self, state: list = None) -> None:
        if state is not None:
            assert len(state) == self.N_STATES, \
                f"Expected {self.N_STATES} states, got {len(state)}"
            self._state = list(float(v) for v in state)
        else:
            self._state = [0.0] * self.N_STATES
        self._step_count = 0

    @property
    def step_count(self) -> int:
        return self._step_count
