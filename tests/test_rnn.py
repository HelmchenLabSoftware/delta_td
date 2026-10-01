"""Tests of the learnable recurrent representation: initializations reproduce the fixed representations, credit
assignment is correct, and the plastic model runs."""

import numpy as np
import pytest

from deltatd.simulation import representations, simulate, tasks
from deltatd.utils import constants as c
from deltatd.utils import ids

STIMULI = ids.STIMULI
N = c.RNN_UNITS_PER_STIMULUS


def features_of(rep: representations.Representation, protocol: tasks.Protocol) -> np.ndarray:
    """Feature vectors of a representation over a protocol, concatenated over trials (n_steps, n_features)."""
    out = []
    for trial in protocol:
        present, onset, _ = trial.arrays(STIMULI)
        out.append(np.stack([rep.step(present[t], onset[t]) for t in range(trial.duration)]))
    return np.concatenate(out)


def frozen_rnn(init: str, **kwargs) -> representations.RecurrentNetwork:
    return representations.RecurrentNetwork(STIMULI, init=init, noise=0.0, step_ratio=0.0, **kwargs)


@pytest.fixture(scope="module")
def two_cs_protocol() -> tasks.Protocol:
    """Two CSs with different ISIs and the US, followed by unreinforced probe trials."""
    return tasks.blocking_protocol(50, 50, 25, n_phase1=2, n_phase2=2)


def test_csc_init_is_the_delay_line_representation(two_cs_protocol):
    rnn = frozen_rnn(ids.CSC)
    reference = representations.CompleteSerialCompound(STIMULI, max_duration=N, gated_by_presence=False)
    assert np.array_equal(features_of(rnn, two_cs_protocol), features_of(reference, two_cs_protocol))


def test_microstimulus_init_reproduces_the_microstimuli(two_cs_protocol):
    rnn = frozen_rnn(ids.MICROSTIMULUS)
    reference = representations.Microstimulus(STIMULI)
    h = features_of(rnn, two_cs_protocol).reshape(-1, len(STIMULI), N)
    x = features_of(reference, two_cs_protocol).reshape(-1, len(STIMULI), c.N_MICROSTIMULI)
    # Feature maximum is 0.4; the ridge RNN_MS_RIDGE = 1e-2 trades reproduction accuracy (0.06) for a block that is
    # stable under exact credit (docs/model_notes.md); the unregularized fit reproduces to 2e-3.
    assert np.abs(h[:, :, : c.N_MICROSTIMULI] - x).max() < 0.07
    block_exact, _ = representations.microstimulus_block(N, ridge=1e-6)
    exact = frozen_rnn(ids.MICROSTIMULUS)
    for k in range(len(STIMULI)):
        exact.W[k * N : (k + 1) * N, k * N : (k + 1) * N] = block_exact
    h_exact = features_of(exact, two_cs_protocol).reshape(-1, len(STIMULI), N)
    assert np.abs(h_exact[:, :, : c.N_MICROSTIMULI] - x).max() < 2e-3
    hidden = c.N_MICROSTIMULI * c.RNN_MS_EMBEDDING_ORDER
    assert h[:, :, c.N_MICROSTIMULI : hidden].max() <= c.RNN_MS_HIDDEN_SCALE * x.max() + 1e-6
    assert np.all(h[:, :, hidden:] == 0.0)  # the remaining units are silent without initialization noise


def test_presence_init_reproduces_the_presence_representation_on_reinforced_trials(two_cs_protocol):
    rnn = frozen_rnn(ids.PRESENCE)
    reference = representations.Presence(STIMULI)
    h = features_of(rnn, two_cs_protocol).reshape(-1, len(STIMULI), N)
    x = features_of(reference, two_cs_protocol)
    n_reinforced = sum(1 for trial in two_cs_protocol if trial.us_time is not None) * c.TRIAL_DURATION
    assert np.array_equal(h[:n_reinforced, :, 0], x[:n_reinforced])  # the US switches the CS units off exactly
    assert np.all(h[:, :, 1:] == 0.0)
    # Without a US the self-sustaining CS units stay on until the next US (the CS offset is not an input)
    first_probe = next(j for j, trial in enumerate(two_cs_protocol) if trial.us_time is None)
    assert h[(first_probe + 1) * c.TRIAL_DURATION - 1, 0, 0] == pytest.approx(c.PRESENCE_SALIENCE)


@pytest.mark.parametrize("init", [ids.CSC, ids.PRESENCE, ids.MICROSTIMULUS])
def test_frozen_rnn_models_learn_like_their_fixed_representation(init):
    """With the recurrent plasticity off, each initialization yields the value time courses of the fixed model."""
    protocol = tasks.acquisition_protocol(25, n_trials=50)
    rnn = simulate.build_model(ids.RNN_INIT_MODELS[init], representation_kwargs={"noise": 0.0, "step_ratio": 0.0})
    csc_kwargs = {"gated_by_presence": False, "max_duration": N}  # the delay line of the network
    reference = simulate.build_model(init, representation_kwargs=csc_kwargs if init == ids.CSC else {})
    value_rnn = simulate.run_protocol(rnn, protocol)[ids.VALUE]
    value_ref = simulate.run_protocol(reference, protocol)[ids.VALUE]
    tolerance = 0.1 if init == ids.MICROSTIMULUS else 1e-6  # ridge-regularized block (8%) and weights on the copy units
    assert np.abs(value_rnn - value_ref).max() < tolerance


def small_random_rnn(credit: str) -> representations.RecurrentNetwork:
    return representations.RecurrentNetwork(("A", "US"), n_units=2, init="random", credit=credit, horizon=20, seed=1)


ONSETS = [np.array([True, False])] + [np.array([False, False])] * 9 + [np.array([False, True])] + [np.array([False, False])] * 4


def test_exact_sensitivity_matches_finite_differences():
    rnn = small_random_rnn(ids.RNN_EXACT_CREDIT)
    readout = np.random.default_rng(3).standard_normal(rnn.n_features)
    last = 12

    def value_at(weights: np.ndarray) -> float:
        probe = small_random_rnn(ids.RNN_EXACT_CREDIT)
        probe.W = weights
        probe.reset()
        for t in range(last + 1):
            h = probe.step(ONSETS[t], ONSETS[t])
        return float(readout @ h)

    for t in range(last + 2):  # after step `last + 1`, the sensitivity refers to V at step `last`
        rnn.step(ONSETS[t], ONSETS[t])
    numerical = np.zeros_like(rnn.W)
    delta = 1e-6
    for i, j in np.ndindex(rnn.W.shape):
        plus, minus = rnn.W.copy(), rnn.W.copy()
        plus[i, j] += delta
        minus[i, j] -= delta
        numerical[i, j] = (value_at(plus) - value_at(minus)) / (2 * delta)
    assert np.abs(rnn.sensitivity(readout) - numerical).max() < 1e-6


def test_local_credit_is_the_first_term_of_the_exact_chain():
    readout = np.random.default_rng(3).standard_normal(4)
    sensitivities = []
    for credit in (ids.RNN_LOCAL_CREDIT, ids.RNN_EXACT_CREDIT):
        rnn = small_random_rnn(credit)
        rnn.horizon = 1  # the exact chain truncated to the direct term
        rnn.reset()
        for t in range(14):
            rnn.step(ONSETS[t], ONSETS[t])
        sensitivities.append(rnn.sensitivity(readout))
    assert np.allclose(*sensitivities)


def test_plastic_rnn_runs_and_records_its_weight_change():
    model = simulate.build_model(ids.RNN)
    results = simulate.run_protocol(model, tasks.acquisition_protocol(25, n_trials=5), record_features_at=(0, 4), feature_metrics=True)
    assert results[ids.FEATURES].shape == (2, c.TRIAL_DURATION, model.representation.n_features)
    assert np.array_equal(results[ids.FEATURE_TRIALS], [0, 4])
    assert np.all(np.isfinite(results[ids.VALUE]))
    assert results[ids.WEIGHT_CHANGE][-1] > results[ids.WEIGHT_CHANGE][0] > 0
    assert np.all(np.isfinite(results[ids.DIMENSIONALITY]))
    frozen = simulate.build_model(ids.RNN, representation_kwargs={"step_ratio": 0.0})
    assert simulate.run_protocol(frozen, tasks.acquisition_protocol(25, n_trials=2))[ids.WEIGHT_CHANGE].max() == 0.0


def test_dimensionality_of_the_fixed_representations():
    protocol = tasks.acquisition_protocol(25, n_trials=1)
    expected = {ids.CSC: 25, ids.PRESENCE: 1}
    for representation, dimensionality in expected.items():
        results = simulate.run_protocol(simulate.build_model(representation), protocol, feature_metrics=True)
        assert results[ids.DIMENSIONALITY][0] == pytest.approx(dimensionality)
    results = simulate.run_protocol(simulate.build_model(ids.MICROSTIMULUS), protocol, feature_metrics=True)
    assert 1 < results[ids.DIMENSIONALITY][0] < 25
