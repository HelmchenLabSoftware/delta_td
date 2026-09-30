"""Temporal stimulus representations.

Each representation maps the stimuli that are present (and those that just turned on) at the current time step onto
a feature vector x_t that is fed to a learner. The three representations of Ludvig et al. (2012) differ in their
degree of temporal generalization: none for the complete serial compound, complete for the presence representation
and intermediate for the microstimuli. The onset representation only signals stimulus onsets and is the input to the
delta-TD learner, which integrates value changes instead of reading the value out of a temporal representation.
"""

from __future__ import annotations

import numpy as np

from deltatd.utils import constants as c


class Representation:
    """Base class of all stimulus representations.

    :param stimuli: identifiers of all stimuli that can occur, defining the stimulus indexing of `step`.
    """

    def __init__(self, stimuli: tuple[str, ...]):
        self.stimuli = tuple(stimuli)
        self.n_stimuli = len(self.stimuli)
        self.reset()

    @property
    def n_features(self) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        """Erase all memory of past stimuli (called once before a protocol, not between trials)."""

    def step(self, present: np.ndarray, onset: np.ndarray) -> np.ndarray:
        """Advance one time step and return the feature vector.

        :param present: boolean array (n_stimuli,) marking the stimuli that are on during this time step.
        :param onset: boolean array (n_stimuli,) marking the stimuli that turned on at this time step.
        :returns: feature vector of length `n_features`.
        """
        raise NotImplementedError

    def feature_labels(self) -> list[str]:
        raise NotImplementedError


class Presence(Representation):
    """One element per stimulus that is on (with a fixed salience) whenever the stimulus is present."""

    def __init__(self, stimuli: tuple[str, ...], salience: float = c.PRESENCE_SALIENCE):
        self.salience = salience
        super().__init__(stimuli)

    @property
    def n_features(self) -> int:
        return self.n_stimuli

    def step(self, present: np.ndarray, onset: np.ndarray) -> np.ndarray:
        return self.salience * present.astype(float)

    def feature_labels(self) -> list[str]:
        return list(self.stimuli)


class CompleteSerialCompound(Representation):
    """A distinct unit element for every time step since stimulus onset.

    :param gated_by_presence: elements are only active while the stimulus is on (Ludvig et al. 2012). With False the
        representation is a tapped delay line started by the onset that keeps ticking for `max_duration` steps
        regardless of the stimulus duration (the CSC of the dopamine models, Montague et al. 1996).
    """

    def __init__(
        self, stimuli: tuple[str, ...], max_duration: int = c.TRIAL_DURATION, gated_by_presence: bool = True
    ):
        self.max_duration = max_duration
        self.gated_by_presence = gated_by_presence
        super().__init__(stimuli)

    @property
    def n_features(self) -> int:
        return self.n_stimuli * self.max_duration

    def reset(self) -> None:
        self.age = np.full(self.n_stimuli, -1, dtype=int)  # Time steps since onset (-1 when not represented)

    def step(self, present: np.ndarray, onset: np.ndarray) -> np.ndarray:
        alive = present if self.gated_by_presence else np.ones_like(present)
        continuing = alive & ~onset & (self.age >= 0)
        self.age = np.where(onset, 0, np.where(continuing, self.age + 1, -1))
        x = np.zeros(self.n_features)
        active = (self.age >= 0) & (self.age < self.max_duration)
        x[np.flatnonzero(active) * self.max_duration + self.age[active]] = 1.0
        return x

    def feature_labels(self) -> list[str]:
        return [f"{s}_t{t}" for s in self.stimuli for t in range(self.max_duration)]


class Microstimulus(Representation):
    """Coarse coding of an exponentially decaying memory trace triggered by stimulus onset (Ludvig et al. 2008).

    Every stimulus (including the US) starts a trace y = 1 at onset that decays as y_{t+1} = d y_t. The trace height
    is encoded by m Gaussian basis functions centred at 1/m, 2/m, ..., 1, and each microstimulus level is the basis
    function value multiplied by the trace height (Eqs. 6-8 of Ludvig et al. 2012).
    """

    def __init__(
        self,
        stimuli: tuple[str, ...],
        n_microstimuli: int = c.N_MICROSTIMULI,
        sigma: float = c.MICROSTIMULUS_WIDTH,
        decay: float = c.MEMORY_DECAY,
    ):
        self.m = n_microstimuli
        self.sigma = sigma
        self.decay = decay
        self.centers = np.arange(1, self.m + 1) / self.m
        super().__init__(stimuli)

    @property
    def n_features(self) -> int:
        return self.n_stimuli * self.m

    def reset(self) -> None:
        self.trace = np.zeros(self.n_stimuli)

    def basis(self, y: np.ndarray) -> np.ndarray:
        """Gaussian basis functions f(y, mu, sigma) evaluated for every trace height y (n_stimuli, m)."""
        return np.exp(-((y[:, None] - self.centers[None, :]) ** 2) / (2 * self.sigma**2)) / np.sqrt(2 * np.pi)

    def step(self, present: np.ndarray, onset: np.ndarray) -> np.ndarray:
        self.trace = np.where(onset, 1.0, self.trace * self.decay)
        return (self.basis(self.trace) * self.trace[:, None]).ravel()

    def feature_labels(self) -> list[str]:
        return [f"{s}_ms{k + 1}" for s in self.stimuli for k in range(self.m)]


class Onset(Representation):
    """Unit pulse at the onset (and optionally the offset) of each stimulus (input of the delta-TD learner).

    Onsets and offsets are the sensory events that can trigger a learned change of the integrated value estimate.
    Whether the US counts as a stimulus is decided by the protocol runner (`run_protocol(us_as_stimulus=...)`).
    """

    def __init__(self, stimuli: tuple[str, ...], include_offsets: bool = c.DELTA_INCLUDE_OFFSETS):
        self.include_offsets = include_offsets
        super().__init__(stimuli)

    @property
    def n_features(self) -> int:
        return self.n_stimuli * (2 if self.include_offsets else 1)

    def reset(self) -> None:
        self.previous = np.zeros(self.n_stimuli, dtype=bool)

    def step(self, present: np.ndarray, onset: np.ndarray) -> np.ndarray:
        offset = self.previous & ~present
        self.previous = present.copy()
        if self.include_offsets:
            return np.concatenate([onset, offset]).astype(float)
        return onset.astype(float)

    def feature_labels(self) -> list[str]:
        labels = [f"{s}_onset" for s in self.stimuli]
        if self.include_offsets:
            labels += [f"{s}_offset" for s in self.stimuli]
        return labels
