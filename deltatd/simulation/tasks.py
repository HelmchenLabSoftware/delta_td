"""Classical conditioning protocols of Ludvig, Sutton & Kehoe (2012).

A protocol is a list of `Trial` objects. Every trial lasts `TRIAL_DURATION` time steps. A conditioned stimulus with
interstimulus interval ISI turns on ISI steps before the US and lasts ISI time steps, i.e. it terminates when the US
arrives (delay conditioning; set `CS_INCLUDES_US_STEP` to keep it on during the US step). When several CSs are
trained in compound they all coterminate, so their onsets differ when their ISIs differ. The earliest CS onset is
fixed at `CS_ONSET_TIME`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from deltatd.utils import constants as c
from deltatd.utils import ids


@dataclass(frozen=True)
class Trial:
    """One trial: which stimuli occur when, and whether a US is delivered.

    :param stimuli: mapping of CS identifier to (onset time step, duration in time steps).
    :param us_time: time step of the US, or None for an unreinforced trial.
    :param us_magnitude: US intensity.
    :param duration: number of time steps of the trial.
    :param probe: whether this is an unreinforced test trial (excluded from learning curves).
    :param label: free-form label (e.g. "A_alone") used to select probe trials when plotting.
    """

    stimuli: dict[str, tuple[int, int]] = field(default_factory=dict)
    us_time: int | None = None
    us_magnitude: float = c.US_MAGNITUDE
    duration: int = c.TRIAL_DURATION
    probe: bool = False
    label: str = ""

    @property
    def cs_onset(self) -> int:
        """Earliest CS onset (reference point for peak times)."""
        return min(onset for onset, _ in self.stimuli.values()) if self.stimuli else 0

    def arrays(self, stimulus_ids: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Expand the trial into time-step arrays.

        :param stimulus_ids: ordering of the stimuli (must contain ids.US and all CS identifiers of the trial).
        :returns: present (duration, n_stimuli) bool, onset (duration, n_stimuli) bool, reward (duration,) float.
        """
        present = np.zeros((self.duration, len(stimulus_ids)), dtype=bool)
        onset = np.zeros_like(present)
        reward = np.zeros(self.duration)
        for cs, (t_on, length) in self.stimuli.items():
            idx = stimulus_ids.index(cs)
            present[t_on : min(t_on + length, self.duration), idx] = True
            onset[t_on, idx] = True
        if self.us_time is not None:
            idx = stimulus_ids.index(ids.US)
            present[self.us_time, idx] = True
            onset[self.us_time, idx] = True
            reward[self.us_time] = self.us_magnitude
        return present, onset, reward


Protocol = list[Trial]


def cs_duration(isi: int) -> int:
    """Default duration of a CS trained with a given ISI (at least one time step)."""
    return max(isi + int(c.CS_INCLUDES_US_STEP), 1)


def delay_trial(
    isis: dict[str, int],
    reinforced: bool = True,
    durations: dict[str, int] | None = None,
    probe: bool = False,
    label: str = "",
) -> Trial:
    """Trial with coterminating CSs.

    :param isis: ISI of each CS. The US (if any) occurs at CS_ONSET_TIME + max(ISI); each CS turns on ISI steps
        before the US and lasts `cs_duration(ISI)` steps unless overridden by `durations`.
    :param reinforced: whether the US is delivered.
    :param durations: optional CS durations overriding the default (e.g. extended CS on probe trials).
    :param probe: mark as probe trial.
    :param label: trial label.
    """
    if not isis:
        raise ValueError("A trial needs at least one CS")
    t_us = c.CS_ONSET_TIME + max(isis.values())
    durations = durations or {}
    stimuli = {cs: (t_us - isi, durations.get(cs, cs_duration(isi))) for cs, isi in isis.items()}
    return Trial(stimuli=stimuli, us_time=t_us if reinforced else None, probe=probe, label=label)


def probe_trials(isis: dict[str, int], durations: dict[str, int] | None = None) -> Protocol:
    """Unreinforced test trials with each CS alone and all CSs in compound.

    :param isis: ISIs (i.e. onsets relative to the omitted US) of the CSs as in the last training phase.
    :param durations: CS durations on the probe trials; defaults to the training durations. Ludvig et al.
        (2012, Fig. 6) show a secondary response peak of the CSC at the phase-1 US time, which requires the probe CS to
        stay on that long, so protocols with an ISI change pass the longest duration a CS was trained with.
    """
    trials = [
        delay_trial({cs: isi}, reinforced=False, durations=durations, probe=True, label=f"{cs}_alone")
        for cs, isi in isis.items()
    ]
    if len(isis) > 1:
        trials.append(delay_trial(isis, reinforced=False, durations=durations, probe=True, label="compound"))
    return trials


def acquisition_protocol(isi: int, n_trials: int = c.N_TRIALS_ACQUISITION) -> Protocol:
    """Simple acquisition of CS A with a fixed ISI (acquisition set, Fig. 2 and 3)."""
    return [delay_trial({ids.CS_A: isi}, label="train") for _ in range(n_trials)]


def timing_protocol(isi: int, n_trials: int = c.N_TRIALS_TIMING, probe_every: int = c.PROBE_EVERY) -> Protocol:
    """Acquisition with every `probe_every`-th trial an unreinforced probe with the CS extended to 2 x ISI (Fig. 4)."""
    trials = []
    for k in range(1, n_trials + 1):
        if k % probe_every == 0:
            trials.append(
                delay_trial(
                    {ids.CS_A: isi}, reinforced=False, durations={ids.CS_A: 2 * isi}, probe=True, label="probe"
                )
            )
        else:
            trials.append(delay_trial({ids.CS_A: isi}, label="train"))
    return trials


def blocking_protocol(
    isi_a_phase1: int,
    isi_a_phase2: int,
    isi_b: int,
    n_phase1: int = c.N_TRIALS_BLOCKING_PHASE,
    n_phase2: int = c.N_TRIALS_BLOCKING_PHASE,
) -> Protocol:
    """Blocking: CS A alone (phase 1), then A + B in compound (phase 2), then probe trials (Fig. 5 and 6).

    On the probe trials each CS stays on for the longest duration it was trained with (relevant when the ISI of A
    changes between phases, see `probe_trials`).
    """
    phase1 = [delay_trial({ids.CS_A: isi_a_phase1}, label="phase1") for _ in range(n_phase1)]
    compound = {ids.CS_A: isi_a_phase2, ids.CS_B: isi_b}
    phase2 = [delay_trial(compound, label="phase2") for _ in range(n_phase2)]
    probe_durations = {ids.CS_A: cs_duration(max(isi_a_phase1, isi_a_phase2)), ids.CS_B: cs_duration(isi_b)}
    return phase1 + phase2 + probe_trials(compound, durations=probe_durations)


def overshadowing_protocol(
    isi_a: int | None, isi_b: int = c.OVERSHADOWING_ISI_B, n_trials: int = c.N_TRIALS_OVERSHADOWING
) -> Protocol:
    """Overshadowing: A + B trained in compound (or B alone when `isi_a` is None), then probe trials (Fig. 7)."""
    compound = {ids.CS_B: isi_b} if isi_a is None else {ids.CS_A: isi_a, ids.CS_B: isi_b}
    return [delay_trial(compound, label="train") for _ in range(n_trials)] + probe_trials(compound)
