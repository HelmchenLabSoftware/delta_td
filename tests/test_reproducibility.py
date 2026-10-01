"""Every random element is seeded: repeated simulations are bit-for-bit identical, and the seed matters."""

import numpy as np

from deltatd.simulation import representations, simulate, tasks
from deltatd.utils import constants as c
from deltatd.utils import ids


def run(model_id: str, n_trials: int = 3) -> dict[str, np.ndarray]:
    return simulate.run_protocol(simulate.build_model(model_id), tasks.acquisition_protocol(25)[:n_trials])


def test_runs_are_identical():
    for model_id in (ids.RNN, ids.RNN_PRESENCE_INIT, ids.RNN_EXACT, ids.CSC, ids.DELTA):
        first, second = run(model_id), run(model_id)
        for key in (ids.VALUE, ids.TD_ERROR, ids.RESPONSE, ids.WEIGHT_CHANGE):
            assert np.array_equal(first[key], second[key], equal_nan=True), (model_id, key)


def test_rnn_initial_weights_depend_only_on_the_seed():
    stimuli = tuple(ids.STIMULI)
    for init in ids.RNN_INITS:
        same = [representations.initial_weights(init, stimuli, c.RNN_UNITS_PER_STIMULUS, c.RNN_INIT_NOISE, c.RNN_RANDOM_SPECTRAL_RADIUS, 0) for _ in range(2)]
        other = representations.initial_weights(init, stimuli, c.RNN_UNITS_PER_STIMULUS, c.RNN_INIT_NOISE, c.RNN_RANDOM_SPECTRAL_RADIUS, 1)
        assert np.array_equal(same[0][0], same[1][0]) and np.array_equal(same[0][1], same[1][1]), init
        assert not np.array_equal(same[0][0], other[0]), init  # the noise / random matrix does change with the seed


def test_partial_reinforcement_schedule_is_seeded():
    schedule = lambda seed: [trial.us_time for trial in tasks.partial_reinforcement_protocol(0.5, n_trials=50, seed=seed)]
    assert schedule(0) == schedule(0)
    assert schedule(0) != schedule(1)
    assert c.DA_PARTIAL_SEED == c.RNN_SEED == c.SEED  # one constant controls everything


def test_global_numpy_rng_is_not_used():
    """Perturbing the global RNG must not change a simulation (all draws come from local generators)."""
    np.random.seed(12345)
    first = run(ids.RNN, n_trials=2)
    np.random.seed(54321)
    np.random.standard_normal(1000)
    second = run(ids.RNN, n_trials=2)
    assert np.array_equal(first[ids.VALUE], second[ids.VALUE])
