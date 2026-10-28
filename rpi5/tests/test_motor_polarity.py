"""Motor wiring polarity in the plant (spec S01, 20-21/10/2026)."""
import numpy as np

from params import MecanumParams
from plant_step import plant_step


def _run(polarity, steps=300, lifted=False):
    p = MecanumParams()
    p.motor_polarity = np.array(polarity, dtype=float)
    if lifted:
        p.update_physical(wheels_lifted=True)
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
    # Wheels lifted: on the ground the body's rotating-frame term M wz x v (bug
    # HIL-PLANT-ROTFRAME, 28/10) is even in the velocity, so the ground response to a
    # reversed drive is not an exact mirror (physically right); the polarity check is not about that.
    w_pos = _run([1, 1, 1, 1], lifted=True)
    w_neg = _run([-1, -1, -1, -1], lifted=True)
    assert np.allclose(w_neg, -w_pos, atol=1e-9)
    g_pos, g_neg = _run([1, 1, 1, 1]), _run([-1, -1, -1, -1])
    assert np.allclose(g_neg, -g_pos, rtol=0.05)       # on the ground: mirror within 5 %
