import numpy as np

from deltatd.simulation import representations
from deltatd.utils import ids

STIMULI = ids.STIMULI


def test_presence_is_salience_while_present():
    rep = representations.Presence(STIMULI, salience=0.2)
    x = rep.step(present=np.array([True, False, False]), onset=np.array([True, False, False]))
    assert np.allclose(x, [0.2, 0.0, 0.0])
    x = rep.step(present=np.array([False, False, False]), onset=np.array([False, False, False]))
    assert np.allclose(x, 0.0)


def test_csc_activates_one_new_element_per_time_step():
    rep = representations.CompleteSerialCompound(STIMULI, max_duration=10)
    assert rep.n_features == 30
    present = np.array([True, False, False])
    seen = []
    for t in range(4):
        x = rep.step(present, onset=present if t == 0 else np.zeros(3, bool))
        assert x.sum() == 1
        seen.append(int(np.flatnonzero(x)[0]))
    assert seen == [0, 1, 2, 3]
    x = rep.step(np.zeros(3, bool), np.zeros(3, bool))
    assert x.sum() == 0


def test_microstimulus_trace_starts_at_one_and_decays():
    rep = representations.Microstimulus(STIMULI, n_microstimuli=6, sigma=0.08, decay=0.985)
    assert rep.n_features == 18
    onset = np.array([True, False, False])
    x0 = rep.step(onset, onset)
    assert np.isclose(rep.trace[0], 1.0)
    assert x0[:6].argmax() == 5  # the last basis function is centred at trace height 1
    assert np.allclose(x0[6:], 0.0)
    for _ in range(200):
        x = rep.step(np.zeros(3, bool), np.zeros(3, bool))
    assert np.isclose(rep.trace[0], 0.985**200)
    assert x[:6].argmax() < 5  # later microstimuli take over as the trace decays


def test_onset_representation_is_a_pulse():
    rep = representations.Onset(STIMULI)
    x = rep.step(present=np.array([True, True, False]), onset=np.array([True, False, False]))
    assert np.allclose(x, [1.0, 0.0, 0.0])
