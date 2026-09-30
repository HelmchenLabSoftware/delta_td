"""Assemble a model (representation + learner + response rule) and run it through a protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from deltatd.simulation import learners, representations, response, tasks
from deltatd.utils import constants as c
from deltatd.utils import ids


@dataclass
class Model:
    representation: representations.Representation
    learner: learners.Learner
    response: response.LeakyIntegratorResponse
    name: str


def build_model(
    representation: ids.RepresentationID,
    stimuli: tuple[str, ...] = ids.STIMULI,
    representation_kwargs: dict[str, Any] | None = None,
    learner_kwargs: dict[str, Any] | None = None,
    response_kwargs: dict[str, Any] | None = None,
) -> Model:
    """Construct the model associated with a representation identifier using the default parameters.

    The three Ludvig representations are combined with TD(lambda); the delta identifier selects the onset
    representation combined with the delta-TD learner.
    """
    representation_kwargs = representation_kwargs or {}
    learner_kwargs = learner_kwargs or {}
    response_kwargs = response_kwargs or {}
    if representation == ids.CSC:
        rep = representations.CompleteSerialCompound(stimuli, **representation_kwargs)
        learner = learners.TDLambda(rep.n_features, **learner_kwargs)
    elif representation == ids.MICROSTIMULUS:
        rep = representations.Microstimulus(stimuli, **representation_kwargs)
        learner = learners.TDLambda(rep.n_features, **learner_kwargs)
    elif representation == ids.PRESENCE:
        rep = representations.Presence(stimuli, **representation_kwargs)
        learner = learners.TDLambda(rep.n_features, **learner_kwargs)
    elif representation == ids.DELTA:
        rep = representations.Onset(stimuli, **representation_kwargs)
        learner = learners.DeltaTD(rep.n_features, **learner_kwargs)
    else:
        raise ValueError(f"Unknown representation '{representation}', expected one of {ids.ALL_REPRESENTATIONS}")
    return Model(rep, learner, response.LeakyIntegratorResponse(**response_kwargs), name=representation)


def run_protocol(model: Model, protocol: tasks.Protocol, us_as_stimulus: bool | None = None) -> dict[str, np.ndarray]:
    """Simulate a protocol trial by trial and record the variables of interest.

    :param model: model to simulate (its state is modified in place).
    :param protocol: list of trials.
    :param us_as_stimulus: whether the US spawns representation elements (Ludvig et al. 2008 assumed it does).
        Defaults to True for the Ludvig representations and to `DELTA_US_AS_STIMULUS` for the delta model.
    :returns: dictionary of arrays keyed by `ids.RecordKey`.
    """
    if us_as_stimulus is None:
        us_as_stimulus = c.DELTA_US_AS_STIMULUS if model.name == ids.DELTA else True
    stimulus_ids = model.representation.stimuli
    us_idx = stimulus_ids.index(ids.US)
    cs_idx = [k for k in range(len(stimulus_ids)) if k != us_idx]
    n_trials = len(protocol)
    duration = max(trial.duration for trial in protocol)
    value = np.full((n_trials, duration), np.nan)
    resp = np.full((n_trials, duration), np.nan)
    td_error = np.full((n_trials, duration), np.nan)
    us_time = np.full(n_trials, -1, dtype=int)
    cs_onset = np.zeros(n_trials, dtype=int)
    probe = np.zeros(n_trials, dtype=bool)
    labels = np.array([trial.label for trial in protocol], dtype=str)

    for j, trial in enumerate(protocol):
        present, onset, reward = trial.arrays(stimulus_ids)
        if not us_as_stimulus:
            present[:, us_idx] = False
            onset[:, us_idx] = False
        cs_present = present[:, cs_idx]
        cs_offset = np.zeros(trial.duration, dtype=bool)
        cs_offset[1:] = (cs_present[:-1] & ~cs_present[1:]).any(axis=1)
        model.learner.start_trial()
        model.response.start_trial()
        for t in range(trial.duration):
            x = model.representation.step(present[t], onset[t])
            out = model.learner.step(
                x,
                reward[t],
                us_present=trial.us_time == t,
                cs_offset=bool(cs_offset[t]),
                trial_end=t == trial.duration - 1,
            )
            value[j, t] = out.value
            td_error[j, t] = out.td_error
            resp[j, t] = model.response.step(out.value)
        us_time[j] = -1 if trial.us_time is None else trial.us_time
        cs_onset[j] = trial.cs_onset
        probe[j] = trial.probe

    cr_level = np.nanmax(resp, axis=1)
    peak_time = np.nanargmax(np.nan_to_num(resp, nan=-np.inf), axis=1) - cs_onset
    peak_time = np.where(cr_level > 0, peak_time, np.nan)
    return {
        ids.VALUE: value.astype(np.float32),
        ids.RESPONSE: resp.astype(np.float32),
        ids.TD_ERROR: td_error.astype(np.float32),
        ids.CR_LEVEL: cr_level,
        ids.PEAK_TIME: peak_time,
        ids.PROBE: probe,
        ids.LABEL: labels,
        ids.US_TIME: us_time,
        ids.CS_ONSET: cs_onset,
    }


def save_results(path, results: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **results)


def load_results(path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
