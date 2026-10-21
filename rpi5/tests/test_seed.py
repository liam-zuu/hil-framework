"""Seeded plant runs are reproducible (THESIS review 20/10/2026, item b)."""
import copy
import numpy as np

from params import MecanumParams
from mecanum_plant import MecanumPlant


def _params_with_slip():
    p = MecanumParams()
    p.slip.enabled = True
    p.slip.prob_spontaneous = 0.05
    return p


def _run(plant, cmds):
    return [plant.step(c) for c in cmds]


def _cmds(n=500, seed=123):
    r = np.random.default_rng(seed)
    return [list(r.uniform(-0.8, 0.8, 4)) for _ in range(n)]


def test_same_seed_same_trajectory():
    a = _run(MecanumPlant(_params_with_slip(), seed=7), _cmds())
    b = _run(MecanumPlant(_params_with_slip(), seed=7), _cmds())
    assert np.array_equal(np.array(a), np.array(b))


def test_reset_restarts_the_random_stream():
    p = MecanumPlant(_params_with_slip(), seed=7)
    a = _run(p, _cmds())
    p.reset()
    b = _run(p, _cmds())
    assert np.array_equal(np.array(a), np.array(b))


def test_random_draws_do_not_depend_on_commands():
    # Same seed, different commands: the slip random stream stays aligned, so a
    # zero-command plant and a driven plant consume the same number of draws.
    p1 = MecanumPlant(_params_with_slip(), seed=3)
    p2 = MecanumPlant(_params_with_slip(), seed=3)
    _run(p1, [[0, 0, 0, 0]] * 200)
    _run(p2, _cmds(200))
    assert p1._rng.random() == p2._rng.random()


def test_default_seed_is_fixed():
    assert MecanumPlant().seed == 0
