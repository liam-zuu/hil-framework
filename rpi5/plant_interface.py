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


class OmegaProfilePlant(MockPlant):
    """
    Wheel speeds from a fixed profile, no dynamics (run_hil.py --omega-steps).

    For checking EncoderGen with speed changes inside one run (bug H7-ENC-RETIME,
    28/10/2026): --fixed-omega only holds one set of speeds. State uses the frame
    order, like MockPlant: [omega1..omega4, 0 x 6].

    segments: list of (omegas[4], duration_s, ramp). ramp False: hold the speeds;
    True: go linearly from the previous segment's speeds (0 before the first) to
    these over the segment. Played once, then all wheels 0.
    """

    def __init__(self, segments, dt: float = 0.001):
        super().__init__()
        self._seg = []
        k, prev = 0, [0.0] * 4
        for omegas, duration_s, ramp in segments:
            n = max(1, int(round(duration_s / dt)))
            self._seg.append((k, k + n, list(prev), [float(w) for w in omegas], bool(ramp)))
            k += n
            prev = [float(w) for w in omegas]
        self._n_total = k

    @property
    def total_steps(self) -> int:
        return self._n_total

    def step(self, torques: list) -> list:
        k = self._step_count
        self._step_count += 1
        w = [0.0] * 4
        for a, b, w0, w1, ramp in self._seg:
            if a <= k < b:
                if ramp:
                    f = (k - a + 1) / (b - a)
                    w = [x + (y - x) * f for x, y in zip(w0, w1)]
                else:
                    w = w1
                break
        self._state[0:4] = w
        return list(self._state)
