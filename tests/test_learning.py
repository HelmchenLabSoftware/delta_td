import numpy as np

from deltatd.simulation import simulate, tasks
from deltatd.utils import constants as c
from deltatd.utils import ids


def acquisition(representation, isi=25, n_trials=200, **kwargs):
    model = simulate.build_model(representation, **kwargs)
    return simulate.run_protocol(model, tasks.acquisition_protocol(isi, n_trials)), model


def test_csc_learns_the_discounted_return_ramp():
    results, _ = acquisition(ids.CSC)
    value = results[ids.VALUE][-1]
    t_us = results[ids.US_TIME][-1]
    for k in (1, 5, 10):
        assert np.isclose(value[t_us - k], c.DISCOUNT ** (k - 1), atol=0.05)
    assert np.isclose(value[t_us], 0.0, atol=0.05)


def test_presence_converges_to_a_flat_intermediate_prediction():
    results, _ = acquisition(ids.PRESENCE)
    value = results[ids.VALUE][-1]
    onset, t_us = results[ids.CS_ONSET][-1], results[ids.US_TIME][-1]
    assert 0.5 < value[onset] < 0.8
    assert np.ptp(value[onset : t_us - 1]) < 0.05


def test_microstimulus_prediction_peaks_near_us():
    results, _ = acquisition(ids.MICROSTIMULUS)
    value = results[ids.VALUE][-1]
    onset, t_us = results[ids.CS_ONSET][-1], results[ids.US_TIME][-1]
    assert abs(int(np.argmax(value[onset : t_us + 5])) + onset - (t_us - 1)) <= 3
    assert value[t_us - 1] > 0.7


def test_delta_td_learns_the_us_magnitude_as_a_step_and_cancels_it_at_the_us():
    results, model = acquisition(ids.DELTA, learner_kwargs={"gamma": 1.0, "reset_rule": ids.RESET_AT_TRIAL_END})
    value = results[ids.VALUE][-1]
    onset, t_us = results[ids.CS_ONSET][-1], results[ids.US_TIME][-1]
    assert np.isclose(value[onset], c.US_MAGNITUDE, atol=0.02)
    assert np.allclose(value[onset:t_us], value[onset], atol=1e-6)  # gamma = 1: no growth before the US
    assert np.isclose(value[t_us], 0.0, atol=0.02)  # the learned weight of the US onset event cancels the prediction
    labels = model.representation.feature_labels()
    assert np.isclose(model.learner.w[labels.index("A_onset")], c.US_MAGNITUDE, atol=0.02)
    assert np.isclose(model.learner.w[labels.index("US_onset")], -c.US_MAGNITUDE, atol=0.02)


def test_delta_td_default_reproduces_the_ramp_and_is_bounded_on_probe_trials():
    """Default: gamma as Ludvig, CS offset consumes the prediction. Ramp on reinforced trials, no explosion without US."""
    model = simulate.build_model(ids.DELTA)
    results = simulate.run_protocol(model, tasks.timing_protocol(50, n_trials=500))
    trained = np.flatnonzero(~results[ids.PROBE])[-1]
    value = results[ids.VALUE][trained]
    onset, t_us = results[ids.CS_ONSET][trained], results[ids.US_TIME][trained]
    # Exponential ramp with the discount factor (the unreinforced probes lower the asymptote below the US magnitude)
    assert np.isclose(value[t_us] / value[onset], c.DISCOUNT**-50, rtol=1e-3)
    assert 0.5 < value[t_us] <= c.US_MAGNITUDE
    assert np.isclose(value[t_us + 1], 0.0)  # consumed when the CS terminates at the US
    probe = np.flatnonzero(results[ids.PROBE])[-1]
    assert np.nanmax(np.abs(results[ids.VALUE][probe])) < c.US_MAGNITUDE / c.DISCOUNT**100 + 0.1
    assert np.isclose(results[ids.VALUE][probe][onset + 101], 0.0, atol=1e-6)  # consumed at the CS offset (2 x ISI)
    assert np.all(np.isfinite(model.learner.w))


def test_delta_td_hard_coded_us_reset_consumes_the_prediction():
    results, model = acquisition(ids.DELTA, learner_kwargs={"reset_rule": ids.RESET_ON_US})
    value = results[ids.VALUE][-1]
    t_us = results[ids.US_TIME][-1]
    assert np.isclose(value[t_us], c.US_MAGNITUDE, atol=0.02)
    assert np.isclose(value[t_us + 1], 0.0)
    assert np.isclose(model.learner.w[model.representation.feature_labels().index("US_onset")], 0.0)


def test_delta_td_with_discounting_grows_by_inverse_gamma_and_reproduces_the_ramp():
    results, _ = acquisition(ids.DELTA, n_trials=200, learner_kwargs={"gamma": 0.97, "reset_rule": ids.RESET_AT_TRIAL_END})
    value = results[ids.VALUE][-1]
    onset, t_us = results[ids.CS_ONSET][-1], results[ids.US_TIME][-1]
    assert np.isclose(value[onset + 1] / value[onset], 1 / 0.97)
    assert np.isclose(value[t_us - 1], c.US_MAGNITUDE, atol=0.05)
    assert np.isclose(value[onset], 0.97**24, atol=0.05)


def test_delta_td_offsets_are_separate_events():
    results, model = acquisition(ids.DELTA, n_trials=5, representation_kwargs={"include_offsets": True})
    assert model.representation.n_features == 2 * len(ids.STIMULI)
    assert results[ids.VALUE].shape[1] == c.TRIAL_DURATION


def test_recorded_arrays_have_consistent_shapes():
    results, _ = acquisition(ids.CSC, n_trials=3)
    n_trials, duration = results[ids.VALUE].shape
    assert (n_trials, duration) == (3, c.TRIAL_DURATION)
    for key in (ids.RESPONSE, ids.TD_ERROR):
        assert results[key].shape == (n_trials, duration)
    for key in (ids.CR_LEVEL, ids.PEAK_TIME, ids.PROBE, ids.LABEL, ids.US_TIME, ids.CS_ONSET):
        assert results[key].shape == (n_trials,)
