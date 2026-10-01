"""Assemble a model (representation + learner + response rule) and run it through a protocol."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from deltatd.simulation import learners, representations, response, tasks
from deltatd.utils import constants as c
from deltatd.utils import ids


DIVERGENCE_LIMIT: float = 1e6  # A model whose value exceeds this is stopped, the rest of its results are NaN


@dataclass
class Model:
    representation: representations.Representation
    learner: learners.Learner
    response: response.LeakyIntegratorResponse
    name: str


DOPAMINE_PARAMETERS: dict[str, dict[str, dict[str, Any]]] = {
    ids.CSC: {
        "representation": {"max_duration": c.DA_CSC_MAX_DURATION, "gated_by_presence": False},
        "learner": {"alpha": c.DA_STEP_SIZE, "gamma": c.DA_DISCOUNT, "lam": c.DA_TRACE_DECAY},
    },
    ids.MICROSTIMULUS: {
        "representation": {
            "n_microstimuli": c.DA_N_MICROSTIMULI,
            "sigma": c.DA_MICROSTIMULUS_WIDTH,
            "decay": c.DA_MEMORY_DECAY,
        },
        "learner": {"alpha": c.DA_STEP_SIZE, "gamma": c.DA_DISCOUNT, "lam": c.DA_TRACE_DECAY},
    },
    ids.PRESENCE: {
        "representation": {},
        "learner": {"alpha": c.DA_STEP_SIZE, "gamma": c.DA_DISCOUNT, "lam": c.DA_TRACE_DECAY},
    },
    ids.DELTA: {
        "representation": {},
        "learner": {"eta": c.DA_STEP_SIZE, "gamma": c.DA_DISCOUNT},
    },
    ids.RNN: {
        "representation": {},
        "learner": {"alpha": c.DA_STEP_SIZE, "gamma": c.DA_DISCOUNT, "lam": c.DA_TRACE_DECAY},
    },
}


def build_dopamine_model(
    model: str,
    representation_kwargs: dict[str, Any] | None = None,
    learner_kwargs: dict[str, Any] | None = None,
    seed: int | None = None,
) -> Model:
    """Model with the parameters of Ludvig et al. (2008) (see `DOPAMINE_PARAMETERS`).

    :param model: a representation id or a model variant id (`ids.MODEL_VARIANTS`).
    :param representation_kwargs, learner_kwargs: overrides applied on top of the 2008 parameters.
    :param seed: seed of a stochastic model (see `build_model`).
    """
    parameters = DOPAMINE_PARAMETERS[ids.base_representation(model)]
    return build_model(
        model,
        representation_kwargs={**parameters["representation"], **(representation_kwargs or {})},
        learner_kwargs={**parameters["learner"], **(learner_kwargs or {})},
        seed=seed,
    )


def is_stochastic(model: str) -> bool:
    """Whether a model's construction draws random numbers (only the recurrent network does: its initial weights)."""
    return ids.base_representation(model) == ids.RNN




def build_model(
    model: str,
    stimuli: tuple[str, ...] = ids.STIMULI,
    representation_kwargs: dict[str, Any] | None = None,
    learner_kwargs: dict[str, Any] | None = None,
    response_kwargs: dict[str, Any] | None = None,
    seed: int | None = None,
) -> Model:
    """Construct the model associated with a representation or model variant identifier (default 2012 parameters).

    The three Ludvig representations are combined with TD(lambda); the delta identifier selects the onset
    representation combined with the delta-TD learner; the rnn identifier selects the plastic recurrent network
    (whose recurrent plasticity is coupled to the parameters of its TD(lambda) readout). A model variant
    (`ids.MODEL_VARIANTS`) applies its overrides before the explicit keyword arguments.

    :param seed: seed of the random elements of a stochastic model (`is_stochastic`); None keeps the default
        `constants.SEED`. Deterministic models ignore it.
    """
    representation = ids.base_representation(model)
    variant = ids.MODEL_VARIANTS.get(model, ids.ModelVariant(representation))
    representation_kwargs = {**variant.representation_kwargs, **(representation_kwargs or {})}
    if seed is not None and is_stochastic(model):
        representation_kwargs = {**representation_kwargs, "seed": seed}
    learner_kwargs = {**variant.learner_kwargs, **(learner_kwargs or {})}
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
    elif representation == ids.RNN:
        rep = representations.RecurrentNetwork(stimuli, **representation_kwargs)
        learner = learners.TDLambda(rep.n_features, **learner_kwargs)
        rep.couple(learner.alpha, learner.gamma, learner.lam)
    else:
        raise ValueError(f"Unknown representation '{representation}', expected one of {ids.ALL_REPRESENTATIONS}")
    return Model(rep, learner, response.LeakyIntegratorResponse(**response_kwargs), name=model)


MODEL_FACTORIES = {"default": build_model, "dopamine": build_dopamine_model}
"""Model builders by name (picklable reference for the parallel simulation jobs)."""


def participation_ratio(x: np.ndarray) -> float:
    """Effective dimensionality (sum s_i)^2 / sum s_i^2 of the eigenvalues s_i of the Gram matrix of the rows of x."""
    eigenvalues = np.linalg.svd(x, compute_uv=False) ** 2
    total = eigenvalues.sum()
    return float(total**2 / (eigenvalues**2).sum()) if total > 0 else np.nan


def run_protocol(
    model: Model,
    protocol: tasks.Protocol,
    us_as_stimulus: bool | None = None,
    record_features_at: tuple[int, ...] = (),
    feature_metrics: bool = False,
) -> dict[str, np.ndarray]:
    """Simulate a protocol trial by trial and record the variables of interest.

    :param model: model to simulate (its state is modified in place).
    :param protocol: list of trials.
    :param us_as_stimulus: whether the US spawns representation elements (Ludvig et al. 2008 assumed it does).
        Defaults to True for the Ludvig representations and to `DELTA_US_AS_STIMULUS` for the delta model.
    :param record_features_at: 0-based indices of the trials whose feature vectors are stored (`ids.FEATURES`).
    :param feature_metrics: compute the dimensionality of the feature trajectory during the CS of every trial.
    :returns: dictionary of arrays keyed by `ids.RecordKey`.
    """
    if us_as_stimulus is None:
        us_as_stimulus = c.DELTA_US_AS_STIMULUS if ids.base_representation(model.name) == ids.DELTA else True
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
    dimensionality = np.full(n_trials, np.nan)
    weight_change = np.zeros(n_trials)
    record_features_at = tuple(sorted(set(record_features_at)))
    n_features = model.representation.n_features
    features = np.zeros((len(record_features_at), duration, n_features), dtype=np.float32)
    trial_features = np.zeros((duration, n_features)) if (feature_metrics or record_features_at) else None

    diverged = False
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
            model.representation.learn(out.td_error, model.learner.w)
            value[j, t] = out.value
            td_error[j, t] = out.td_error
            resp[j, t] = model.response.step(out.value)
            if trial_features is not None:
                trial_features[t] = x
            if not np.isfinite(out.value) or abs(out.value) > DIVERGENCE_LIMIT:
                print(f"    {model.name}: value diverged on trial {j + 1}, remaining trials left as NaN", flush=True)
                diverged = True
                break
        if diverged:
            break
        us_time[j] = -1 if trial.us_time is None else trial.us_time
        cs_onset[j] = trial.cs_onset
        probe[j] = trial.probe
        weight_change[j] = model.representation.weight_change()
        if not np.isfinite(weight_change[j]):  # NaN weights silence a rectified network instead of blowing it up
            print(f"    {model.name}: weights diverged on trial {j + 1}, remaining trials left as NaN", flush=True)
            value[j:] = np.nan
            break
        if j in record_features_at:
            features[record_features_at.index(j)] = trial_features
        if feature_metrics:
            cs_end = trial.us_time if trial.us_time is not None else max(on + n for on, n in trial.stimuli.values())
            dimensionality[j] = participation_ratio(trial_features[trial.cs_onset : cs_end])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN trials after a divergence
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
        ids.DIMENSIONALITY: dimensionality,
        ids.WEIGHT_CHANGE: weight_change,
        ids.FEATURES: features,
        ids.FEATURE_TRIALS: np.array(record_features_at, dtype=int),
    }


def save_results(path, results: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **results)


def load_results(path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
