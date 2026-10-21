"""Motor wiring polarity in the plant (spec S01, 20-21/10/2026)."""
import numpy as np

from params import MecanumParams
from plant_step import plant_step


def _run(polarity, steps=300):
    p = MecanumParams()
    p.motor_polarity = np.array(polarity, dtype=float)
    x = np.zeros(10)
    u = np.array([0.5, 0.5, 0.5, 0.5])          # duty (dc model)
    rng = np.random.default_rng(0)
    for _ in range(steps):
        x = plant_step(x, u, p, 0.001, rng)
    return x[6:10]


def test_default_polarity_matches_s01_run01():
    # S01 run01: all four wheels turn backward for +PWM with ESP32 motor_dir = +1
    assert np.allclose(MecanumParams().motor_polarity, -1.0)


def test_reversed_wheel_turns_the_other_way():
    w_pos = _run([1, 1, 1, 1])
    w_mix = _run([-1, 1, 1, 1])
    assert w_pos[0] > 0 and w_mix[0] < 0


def test_all_reversed_mirrors_all_wheels():
    w_pos = _run([1, 1, 1, 1])
    w_neg = _run([-1, -1, -1, -1])
    assert np.allclose(w_neg, -w_pos, atol=1e-9)
