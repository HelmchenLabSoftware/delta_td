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


# --- Dopamine experiments of Ludvig, Sutton & Kehoe (2008) -------------------------------------------------------


def cue_trial(
    cues: dict[str, int],
    reward_time: int | None = None,
    reward_magnitude: float = c.US_MAGNITUDE,
    cue_durations: dict[str, int] | None = None,
    duration: int = c.DA_TRIAL_DURATION,
    probe: bool = False,
    label: str = "",
) -> Trial:
    """Trial of the dopamine experiments: cues at given times, optional reward, 500-step trial.

    :param cues: onset time step of each cue (relative to the trial start).
    :param reward_time: time step of the reward, or None.
    :param cue_durations: duration of each cue; by default a cue lasts until the usual reward time
        (`DA_CUE_TIME + DA_REWARD_DELAY`, or the multiple-cue reward time if the cue starts later) when
        `DA_CUE_LASTS_UNTIL_REWARD` is set, and one time step otherwise.
    """
    cue_durations = cue_durations or {}
    stimuli = {}
    for cue, t_on in cues.items():
        if cue in cue_durations:
            length = cue_durations[cue]
        elif c.DA_CUE_LASTS_UNTIL_REWARD:
            usual_reward = c.DA_CUE_TIME + (c.DA_MULTI_REWARD_DELAY if len(cues) > 1 else c.DA_REWARD_DELAY)
            length = max(usual_reward - t_on, 1)
        else:
            length = 1
        stimuli[cue] = (t_on, length)
    return Trial(
        stimuli=stimuli,
        us_time=reward_time,
        us_magnitude=reward_magnitude,
        duration=duration,
        probe=probe,
        label=label,
    )


def dopamine_acquisition_protocol(n_trials: int = c.DA_N_TRIALS) -> Protocol:
    """Cue at time 0, reward exactly 1 s later, on every trial (2008 Fig. 3)."""
    t_reward = c.DA_CUE_TIME + c.DA_REWARD_DELAY
    return [cue_trial({ids.CS_A: c.DA_CUE_TIME}, t_reward, label="train") for _ in range(n_trials)]


def reward_omission_protocol(n_trials: int = c.DA_N_TRIALS) -> Protocol:
    """Acquisition with the reward omitted on the last trial (2008 Fig. 4)."""
    protocol = dopamine_acquisition_protocol(n_trials - 1)
    protocol.append(cue_trial({ids.CS_A: c.DA_CUE_TIME}, None, probe=True, label="omission"))
    return protocol


def partial_reinforcement_protocol(
    probability: float, n_trials: int = c.DA_N_TRIALS_PARTIAL, seed: int = c.DA_PARTIAL_SEED
) -> Protocol:
    """Reward with a fixed probability, then one rewarded and one omission test trial (2008 Fig. 6)."""
    rng = np.random.default_rng(seed)
    t_reward = c.DA_CUE_TIME + c.DA_REWARD_DELAY
    rewarded = rng.random(n_trials) < probability
    protocol = [
        cue_trial({ids.CS_A: c.DA_CUE_TIME}, t_reward if r else None, label="train") for r in rewarded
    ]
    protocol.append(cue_trial({ids.CS_A: c.DA_CUE_TIME}, t_reward, probe=True, label="rewarded"))
    protocol.append(cue_trial({ids.CS_A: c.DA_CUE_TIME}, None, probe=True, label="omission"))
    return protocol


def early_reward_protocol(n_trials: int = c.DA_N_TRIALS, n_probes: int = c.DA_N_EARLY_PROBES) -> Protocol:
    """Acquisition followed by probe trials with the reward 0.5 s instead of 1 s after the cue (2008 Fig. 7)."""
    protocol = dopamine_acquisition_protocol(n_trials)
    t_early = c.DA_CUE_TIME + c.DA_EARLY_REWARD_DELAY
    protocol += [cue_trial({ids.CS_A: c.DA_CUE_TIME}, t_early, probe=True, label="early") for _ in range(n_probes)]
    return protocol


def multiple_cues_protocol(n_trials: int = c.DA_N_TRIALS, test_after: tuple[int, ...] = c.DA_MULTI_EXAMPLE_TRIALS) -> Protocol:
    """Two sequential cues (0 s and 2 s) before the reward (3 s); after `test_after` trials, one test trial with both
    cues and one with the second cue omitted are inserted (2008 Fig. 8)."""
    t_second = c.DA_CUE_TIME + c.DA_SECOND_CUE_DELAY
    t_reward = c.DA_CUE_TIME + c.DA_MULTI_REWARD_DELAY
    both = {ids.CS_A: c.DA_CUE_TIME, ids.CS_B: t_second}
    protocol = []
    for k in range(1, n_trials + 1):
        protocol.append(cue_trial(both, t_reward, label="train"))
        if k in test_after:
            protocol.append(cue_trial(both, t_reward, probe=True, label=f"both_{k}"))
            protocol.append(
                cue_trial(
                    {ids.CS_A: c.DA_CUE_TIME},
                    t_reward,
                    cue_durations={ids.CS_A: both_duration(c.DA_CUE_TIME, t_reward)},
                    probe=True,
                    label=f"omitted_{k}",
                )
            )
    return protocol


def both_duration(t_on: int, t_reward: int) -> int:
    """Cue duration when the cue lasts until the reward (or one step)."""
    return max(t_reward - t_on, 1) if c.DA_CUE_LASTS_UNTIL_REWARD else 1
