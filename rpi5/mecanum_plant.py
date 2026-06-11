"""
MecanumPlant — concrete PlantInterface wrapping plant_step.py

If you already have the Python plant in a different file (the one with
24/24 tests passing), update the import below and adjust the call
signature to match.  The wrapper logic stays identical.
"""
from __future__ import annotations

import numpy as np

from plant_interface import PlantInterface
from plant_step import plant_step
from params import MecanumParams, PARAMS


class MecanumPlant(PlantInterface):
    """
    Physical Mecanum AGV plant wrapping plant_step().

    One instance per simulation run.  Create a new instance (or call
    reset()) between runs to clear internal state — there are no
    persistent module-level variables in plant_step.py.

    Args:
        params: MecanumParams instance.  Defaults to module singleton PARAMS.
        dt:     Timestep [s].  Must match HILConfig.period_us / 1e6.
        seed:   RNG seed for reproducible wheel-slip noise.
                None → non-reproducible (production mode).
    """

    def __init__(
        self,
        params: MecanumParams = PARAMS,
        dt:     float         = 0.001,
        seed:   int           = None,
    ):
        self._params = params
        self._dt     = dt
        self._state  = np.zeros(10, dtype=float)
        self._rng    = np.random.default_rng(seed)
        self._ticks  = 0

    # ── PlantInterface ────────────────────────────────────────────────────

    def step(self, torques: list) -> list:
        """
        Advance plant by one timestep.

        Args:
            torques: 4-element list/array [τ1, τ2, τ3, τ4] in N·m.

        Returns:
            Updated 10-element state as plain Python list.
        """
        tau = np.asarray(torques, dtype=float)
        self._state = plant_step(self._state, tau, self._params, self._dt, self._rng)
        self._ticks += 1
        return self._state.tolist()

    def get_state(self) -> list:
        return self._state.tolist()

    def reset(self, state: list = None) -> None:
        if state is not None:
            self._state = np.array(state, dtype=float)
        else:
            self._state = np.zeros(10, dtype=float)
        self._ticks = 0

    # ── Extras ───────────────────────────────────────────────────────────

    @property
    def t(self) -> float:
        """Simulated time elapsed [s]."""
        return self._ticks * self._dt

    @property
    def ticks(self) -> int:
        return self._ticks

    def set_slip(self, enabled: bool) -> None:
        """Enable / disable wheel slip model at runtime."""
        self._params.slip.enabled = enabled
