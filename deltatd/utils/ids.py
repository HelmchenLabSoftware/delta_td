from typing import Literal

# Identifiers of the temporal stimulus representations that are compared
RepresentationID = Literal["csc", "microstimulus", "presence", "delta"]
CSC: RepresentationID = "csc"  # Complete serial compound (one element per stimulus time step)
MICROSTIMULUS: RepresentationID = "microstimulus"  # Coarse-coded decaying memory trace (Ludvig et al. 2008)
PRESENCE: RepresentationID = "presence"  # Single element per stimulus, active while the stimulus is on
DELTA: RepresentationID = "delta"  # Integrated value-change model (Schoenfeld et al. 2024, Supplementary Note 2)
LUDVIG_REPRESENTATIONS: list[RepresentationID] = [CSC, MICROSTIMULUS, PRESENCE]
ALL_REPRESENTATIONS: list[RepresentationID] = LUDVIG_REPRESENTATIONS + [DELTA]

# Named model variants: a base representation with learner overrides, usable wherever a representation id is plotted
ModelVariantID = Literal["delta_offset"]
DELTA_OFFSET: ModelVariantID = "delta_offset"  # Delta-TD whose prediction is only consumed by a CS offset (not by the US)
MODEL_VARIANTS: dict[str, tuple[RepresentationID, dict[str, object]]] = {DELTA_OFFSET: (DELTA, {"reset_rule": "offset"})}
REPRESENTATION_LABELS: dict[str, str] = {
    CSC: "CSC",
    MICROSTIMULUS: "Microstimulus",
    PRESENCE: "Presence",
    DELTA: r"$\Delta$-TD",
    DELTA_OFFSET: r"$\Delta$-TD (only CS offset consumes)",
}


def base_representation(model: str) -> RepresentationID:
    """Representation underlying a representation id or a model variant id."""
    return MODEL_VARIANTS[model][0] if model in MODEL_VARIANTS else model

# Identifiers of the studies whose experiments are reproduced
StudyID = Literal["ludvig2012", "ludvig2008"]
LUDVIG2012: StudyID = "ludvig2012"  # Ludvig, Sutton & Kehoe (2012) Learning & Behavior 40:305-319
LUDVIG2008: StudyID = "ludvig2008"  # Ludvig, Sutton & Kehoe (2008) Neural Computation 20:3034-3054

# Identifiers of the experiments of Ludvig et al. (2012) and (2008)
ExperimentID = Literal[
    "acquisition",
    "timing",
    "blocking",
    "blocking_isi_change",
    "overshadowing",
    "delta_variants",
    "dopamine_acquisition",
    "reward_omission",
    "partial_reinforcement",
    "early_reward",
    "multiple_cues",
]
ACQUISITION: ExperimentID = "acquisition"  # 2012 Fig. 2 and 3
TIMING: ExperimentID = "timing"  # 2012 Fig. 4
BLOCKING: ExperimentID = "blocking"  # 2012 Fig. 5
BLOCKING_ISI_CHANGE: ExperimentID = "blocking_isi_change"  # 2012 Fig. 6
OVERSHADOWING: ExperimentID = "overshadowing"  # 2012 Fig. 7
DELTA_VARIANTS: ExperimentID = "delta_variants"  # Supplementary: delta-TD consumption-rule variants on 2012 tasks
DA_ACQUISITION: ExperimentID = "dopamine_acquisition"  # 2008 Fig. 3
REWARD_OMISSION: ExperimentID = "reward_omission"  # 2008 Fig. 4
PARTIAL_REINFORCEMENT: ExperimentID = "partial_reinforcement"  # 2008 Fig. 6
EARLY_REWARD: ExperimentID = "early_reward"  # 2008 Fig. 7
MULTIPLE_CUES: ExperimentID = "multiple_cues"  # 2008 Fig. 8

# Stimulus identifiers. The unconditioned stimulus is itself a stimulus that can spawn representation elements.
StimulusID = Literal["A", "B", "US"]
CS_A: StimulusID = "A"
CS_B: StimulusID = "B"
US: StimulusID = "US"
STIMULI: tuple[StimulusID, ...] = (CS_A, CS_B, US)

# How the delta-TD model determines the imminent reward prediction R_hat that resets the integrated value estimate.
# In Schoenfeld et al. the reset was tied to the lick action; classical conditioning has no action (open question).
ResetRule = Literal["offset", "event", "us", "trial_end", "none"]
RESET_ON_OFFSET: ResetRule = "offset"  # R_hat_t = V_hat_t only when a CS turns off (the US itself does not cash in)
RESET_ON_EVENT: ResetRule = "event"  # R_hat_t = V_hat_t when a US is delivered, or when a CS turns off without US (default)
RESET_ON_US: ResetRule = "us"  # R_hat_t = V_hat_t whenever a US is delivered (hard-coded reward timing) + trial-end reset
RESET_AT_TRIAL_END: ResetRule = "trial_end"  # V_hat is only reset (with the corresponding TD error) at trial end
RESET_NONE: ResetRule = "none"  # No reset at all: only learned input-event weights (and a leak) can bring V_hat down

DELTA_VARIANT_LABELS: dict[str, str] = {
    RESET_ON_OFFSET: "only CS offset consumes",
    RESET_ON_EVENT: "US or CS offset consumes (default)",
    RESET_ON_US: "US consumes",
    RESET_AT_TRIAL_END: r"trial end only ($\gamma = 1$)",
}

# Keys of the arrays recorded during a simulated protocol
RecordKey = Literal["value", "response", "td_error", "cr_level", "peak_time", "probe", "label", "us_time", "cs_onset"]
VALUE: RecordKey = "value"  # US prediction V_hat at every time step (n_trials, trial_duration)
RESPONSE: RecordKey = "response"  # Conditioned response level at every time step (n_trials, trial_duration)
TD_ERROR: RecordKey = "td_error"  # Prediction error at every time step (n_trials, trial_duration)
CR_LEVEL: RecordKey = "cr_level"  # Maximal response within each trial (n_trials,)
PEAK_TIME: RecordKey = "peak_time"  # Time step (relative to the earliest CS onset) of the maximal response (n_trials,)
PROBE: RecordKey = "probe"  # Whether the trial was an unreinforced probe trial (n_trials,)
LABEL: RecordKey = "label"  # Free-form trial label set by the protocol (n_trials,)
US_TIME: RecordKey = "us_time"  # Time step of the US (-1 if absent) (n_trials,)
CS_ONSET: RecordKey = "cs_onset"  # Earliest CS onset in the trial (n_trials,)
