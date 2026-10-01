import numpy as np

from deltatd.simulation import representations, simulate, tasks
from deltatd.utils import constants as c
from deltatd.utils import ids


def test_delay_line_csc_keeps_ticking_after_a_brief_cue():
    rep = representations.CompleteSerialCompound(ids.STIMULI, max_duration=10, gated_by_presence=False)
    on = np.array([True, False, False])
    off = np.zeros(3, bool)
    rep.step(on, on)
    x = rep.step(off, off)
    assert x[1] == 1.0 and x.sum() == 1
    x = rep.step(off, off)
    assert x[2] == 1.0


def test_cue_trial_conventions():
    trial = tasks.cue_trial({ids.CS_A: c.DA_CUE_TIME}, c.DA_CUE_TIME + c.DA_REWARD_DELAY)
    onset, length = trial.stimuli[ids.CS_A]
    assert onset == c.DA_CUE_TIME
    assert length == (c.DA_REWARD_DELAY if c.DA_CUE_LASTS_UNTIL_REWARD else 1)
    assert trial.duration == c.DA_TRIAL_DURATION


def test_dopamine_protocols_have_expected_test_trials():
    omission = tasks.reward_omission_protocol(n_trials=5)
    assert len(omission) == 5 and omission[-1].probe and omission[-1].us_time is None
    partial = tasks.partial_reinforcement_protocol(0.5, n_trials=20)
    assert [t.label for t in partial[-2:]] == ["rewarded", "omission"]
    assert 0 < sum(t.us_time is not None for t in partial[:-2]) < 20
    assert tasks.partial_reinforcement_protocol(0.5, n_trials=20)[3].us_time == partial[3].us_time  # seeded
    early = tasks.early_reward_protocol(n_trials=3, n_probes=2)
    assert early[-1].us_time == c.DA_CUE_TIME + c.DA_EARLY_REWARD_DELAY and early[-1].probe
    multi = tasks.multiple_cues_protocol(n_trials=3, test_after=(2,))
    labels = [t.label for t in multi]
    assert labels == ["train", "train", "both_2", "omitted_2", "train"]
    assert set(multi[3].stimuli) == {ids.CS_A}


def test_dopamine_models_learn_to_predict_the_reward():
    for representation in c.DA_REPRESENTATIONS:
        model = simulate.build_dopamine_model(representation)
        n_trials = 100 if representation == ids.RNN else 300  # the RNN is slower to simulate
        results = simulate.run_protocol(model, tasks.dopamine_acquisition_protocol(n_trials=n_trials))
        t_reward = results[ids.US_TIME][-1]
        error = results[ids.TD_ERROR]
        assert abs(error[0, t_reward]) > 0.9  # unexpected reward on the first trial
        if representation == ids.RNN:
            # The random network only learns a broad prediction that outlasts the reward: the value before the
            # reward grows, but the bootstrapped error at the reward is not reduced (see docs/model_notes.md)
            assert results[ids.VALUE][-1, t_reward - 1] > 0.1
        else:
            assert abs(error[-1, t_reward]) < 0.5 * abs(error[0, t_reward])  # partly predicted after training
        assert np.all(np.isfinite(model.learner.w))
