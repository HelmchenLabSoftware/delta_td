import numpy as np

from deltatd.simulation import tasks
from deltatd.utils import constants as c
from deltatd.utils import ids


def test_delay_trial_coterminates_with_us():
    trial = tasks.delay_trial({ids.CS_A: 50, ids.CS_B: 25})
    assert trial.us_time == c.CS_ONSET_TIME + 50
    assert trial.stimuli[ids.CS_A] == (c.CS_ONSET_TIME, tasks.cs_duration(50))
    assert trial.stimuli[ids.CS_B] == (c.CS_ONSET_TIME + 25, tasks.cs_duration(25))
    present, onset, reward = trial.arrays(ids.STIMULI)
    assert present[trial.us_time - 1, 0] and present[trial.us_time - 1, 1]  # both CSs on right before the US
    assert present[trial.us_time, 2]
    assert not present[trial.us_time + 1].any()
    assert onset.sum(axis=0).tolist() == [1, 1, 1]
    assert reward.sum() == c.US_MAGNITUDE and reward[trial.us_time] == c.US_MAGNITUDE


def test_timing_protocol_probe_trials_are_extended_and_unreinforced():
    protocol = tasks.timing_protocol(25, n_trials=10, probe_every=5)
    probes = [trial for trial in protocol if trial.probe]
    assert len(probes) == 2
    assert probes[0].us_time is None
    assert probes[0].stimuli[ids.CS_A] == (c.CS_ONSET_TIME, 50)
    assert protocol[0].stimuli[ids.CS_A] == (c.CS_ONSET_TIME, tasks.cs_duration(25))


def test_blocking_protocol_structure():
    protocol = tasks.blocking_protocol(50, 50, 25, n_phase1=3, n_phase2=2)
    assert len(protocol) == 3 + 2 + 3
    assert [trial.label for trial in protocol[-3:]] == ["A_alone", "B_alone", "compound"]
    assert all(trial.probe and trial.us_time is None for trial in protocol[-3:])
    assert set(protocol[3].stimuli) == {ids.CS_A, ids.CS_B}


def test_blocking_probe_cs_keeps_its_longest_training_duration():
    protocol = tasks.blocking_protocol(100, 25, 25, n_phase1=1, n_phase2=1)
    a_alone, b_alone, compound = protocol[-3:]
    d100, d25 = tasks.cs_duration(100), tasks.cs_duration(25)
    assert a_alone.stimuli[ids.CS_A] == (c.CS_ONSET_TIME, d100)  # phase-1 duration although timed as in phase 2
    assert b_alone.stimuli[ids.CS_B] == (c.CS_ONSET_TIME, d25)
    assert compound.stimuli[ids.CS_A][1] == d100 and compound.stimuli[ids.CS_B][1] == d25
    same_isi = tasks.blocking_protocol(50, 50, 50, n_phase1=1, n_phase2=1)
    assert same_isi[-1].stimuli[ids.CS_A] == (c.CS_ONSET_TIME, tasks.cs_duration(50))


def test_overshadowing_without_a():
    protocol = tasks.overshadowing_protocol(None, n_trials=2)
    assert [trial.label for trial in protocol] == ["train", "train", "B_alone"]
    assert set(protocol[0].stimuli) == {ids.CS_B}
    assert np.all([trial.us_time is not None for trial in protocol[:2]])
