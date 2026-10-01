from typing import Literal, NamedTuple

# Identifiers of the temporal stimulus representations that are compared
RepresentationID = Literal["csc", "microstimulus", "presence", "delta", "rnn"]
CSC: RepresentationID = "csc"  # Complete serial compound (one element per stimulus time step)
MICROSTIMULUS: RepresentationID = "microstimulus"  # Coarse-coded decaying memory trace (Ludvig et al. 2008)
PRESENCE: RepresentationID = "presence"  # Single element per stimulus, active while the stimulus is on
DELTA: RepresentationID = "delta"  # Integrated value-change model (Schoenfeld et al. 2024, Supplementary Note 2)
RNN: RepresentationID = "rnn"  # Learnable representation: recurrent network driven by onset pulses, plastic weights
LUDVIG_REPRESENTATIONS: list[RepresentationID] = [CSC, MICROSTIMULUS, PRESENCE]
ALL_REPRESENTATIONS: list[RepresentationID] = LUDVIG_REPRESENTATIONS + [DELTA, RNN]

# Initial recurrent weights of the RNN representation: random, or reproducing one of the fixed representations
RNNInitID = Literal["random", "csc", "microstimulus", "presence"]
RNN_RANDOM_INIT: RNNInitID = "random"
RNN_INITS: tuple[RNNInitID, ...] = (RNN_RANDOM_INIT, CSC, MICROSTIMULUS, PRESENCE)
# How the TD error is assigned to the recurrent synapses: local eligibility traces (e-prop like, default) or the exact
# semi-gradient of the value propagated backwards through the network over a finite horizon
RNNCreditID = Literal["local", "exact"]
RNN_LOCAL_CREDIT: RNNCreditID = "local"
RNN_EXACT_CREDIT: RNNCreditID = "exact"


class ModelVariant(NamedTuple):
    """A base representation with parameter overrides, usable wherever a representation id is plotted."""

    representation: RepresentationID
    learner_kwargs: dict[str, object] = {}
    representation_kwargs: dict[str, object] = {}


ModelVariantID = Literal[
    "delta_offset",
    "rnn_csc",
    "rnn_microstimulus",
    "rnn_presence",
    "rnn_exact",
    "rnn_csc_exact",
    "rnn_microstimulus_exact",
    "rnn_presence_exact",
]
DELTA_OFFSET: ModelVariantID = "delta_offset"  # Delta-TD whose prediction is only consumed by a CS offset (not by the US)
RNN_CSC_INIT: ModelVariantID = "rnn_csc"  # RNN initialized as a delay line (complete serial compound)
RNN_MICROSTIMULUS_INIT: ModelVariantID = "rnn_microstimulus"  # RNN initialized to reproduce the microstimuli
RNN_PRESENCE_INIT: ModelVariantID = "rnn_presence"  # RNN initialized as a self-sustaining presence unit
RNN_EXACT: ModelVariantID = "rnn_exact"  # ... the same four with the exact (backward-propagated) credit assignment
RNN_CSC_INIT_EXACT: ModelVariantID = "rnn_csc_exact"
RNN_MICROSTIMULUS_INIT_EXACT: ModelVariantID = "rnn_microstimulus_exact"
RNN_PRESENCE_INIT_EXACT: ModelVariantID = "rnn_presence_exact"
MODEL_VARIANTS: dict[str, ModelVariant] = {
    DELTA_OFFSET: ModelVariant(DELTA, {"reset_rule": "offset"}),
    RNN_CSC_INIT: ModelVariant(RNN, {}, {"init": CSC}),
    RNN_MICROSTIMULUS_INIT: ModelVariant(RNN, {}, {"init": MICROSTIMULUS}),
    RNN_PRESENCE_INIT: ModelVariant(RNN, {}, {"init": PRESENCE}),
    RNN_EXACT: ModelVariant(RNN, {}, {"credit": RNN_EXACT_CREDIT}),
    RNN_CSC_INIT_EXACT: ModelVariant(RNN, {}, {"init": CSC, "credit": RNN_EXACT_CREDIT}),
    RNN_MICROSTIMULUS_INIT_EXACT: ModelVariant(RNN, {}, {"init": MICROSTIMULUS, "credit": RNN_EXACT_CREDIT}),
    RNN_PRESENCE_INIT_EXACT: ModelVariant(RNN, {}, {"init": PRESENCE, "credit": RNN_EXACT_CREDIT}),
}
RNN_INIT_MODELS: dict[RNNInitID, str] = {  # RNN init -> model id with the local credit assignment (default)
    RNN_RANDOM_INIT: RNN,
    CSC: RNN_CSC_INIT,
    MICROSTIMULUS: RNN_MICROSTIMULUS_INIT,
    PRESENCE: RNN_PRESENCE_INIT,
}
RNN_INIT_MODELS_EXACT: dict[RNNInitID, str] = {  # RNN init -> model id with the exact credit assignment
    RNN_RANDOM_INIT: RNN_EXACT,
    CSC: RNN_CSC_INIT_EXACT,
    MICROSTIMULUS: RNN_MICROSTIMULUS_INIT_EXACT,
    PRESENCE: RNN_PRESENCE_INIT_EXACT,
}
RNN_MODELS_BY_CREDIT: dict[RNNCreditID, dict[RNNInitID, str]] = {
    RNN_LOCAL_CREDIT: RNN_INIT_MODELS,
    RNN_EXACT_CREDIT: RNN_INIT_MODELS_EXACT,
}
RNN_INIT_LABELS: dict[RNNInitID, str] = {
    RNN_RANDOM_INIT: "random init",
    CSC: "CSC init",
    MICROSTIMULUS: "microstimulus init",
    PRESENCE: "presence init",
}
RNN_CREDIT_LABELS: dict[RNNCreditID, str] = {RNN_LOCAL_CREDIT: "local credit", RNN_EXACT_CREDIT: "exact credit"}
REPRESENTATION_LABELS: dict[str, str] = {
    CSC: "CSC",
    MICROSTIMULUS: "Microstimulus",
    PRESENCE: "Presence",
    DELTA: r"$\Delta$-TD",
    DELTA_OFFSET: r"$\Delta$-TD (only CS offset consumes)",
    RNN: "RNN (random init)",
    RNN_CSC_INIT: "RNN (CSC init)",
    RNN_MICROSTIMULUS_INIT: "RNN (microstimulus init)",
    RNN_PRESENCE_INIT: "RNN (presence init)",
    RNN_EXACT: "RNN (random init, exact credit)",
    RNN_CSC_INIT_EXACT: "RNN (CSC init, exact credit)",
    RNN_MICROSTIMULUS_INIT_EXACT: "RNN (microstimulus init, exact credit)",
    RNN_PRESENCE_INIT_EXACT: "RNN (presence init, exact credit)",
}


def base_representation(model: str) -> RepresentationID:
    """Representation underlying a representation id or a model variant id."""
    return MODEL_VARIANTS[model].representation if model in MODEL_VARIANTS else model

# Identifiers of the studies whose experiments are reproduced
StudyID = Literal["ludvig2012", "ludvig2008", "rnn"]
LUDVIG2012: StudyID = "ludvig2012"  # Ludvig, Sutton & Kehoe (2012) Learning & Behavior 40:305-319
LUDVIG2008: StudyID = "ludvig2008"  # Ludvig, Sutton & Kehoe (2008) Neural Computation 20:3034-3054
RNN_STUDY: StudyID = "rnn"  # Analyses of the learnable RNN representation (initializations, recurrent step size)

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
    "rnn_initialization",
    "rnn_step_size",
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
RNN_INITIALIZATION: ExperimentID = "rnn_initialization"  # Evolution of the RNN representation from each initialization
RNN_STEP_SIZE: ExperimentID = "rnn_step_size"  # Scan of the recurrent step size relative to the readout step size

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
RecordKey = Literal[
    "value",
    "response",
    "td_error",
    "cr_level",
    "peak_time",
    "probe",
    "label",
    "us_time",
    "cs_onset",
    "dimensionality",
    "weight_change",
    "features",
    "feature_trials",
]
VALUE: RecordKey = "value"  # US prediction V_hat at every time step (n_trials, trial_duration)
RESPONSE: RecordKey = "response"  # Conditioned response level at every time step (n_trials, trial_duration)
TD_ERROR: RecordKey = "td_error"  # Prediction error at every time step (n_trials, trial_duration)
CR_LEVEL: RecordKey = "cr_level"  # Maximal response within each trial (n_trials,)
PEAK_TIME: RecordKey = "peak_time"  # Time step (relative to the earliest CS onset) of the maximal response (n_trials,)
PROBE: RecordKey = "probe"  # Whether the trial was an unreinforced probe trial (n_trials,)
LABEL: RecordKey = "label"  # Free-form trial label set by the protocol (n_trials,)
US_TIME: RecordKey = "us_time"  # Time step of the US (-1 if absent) (n_trials,)
CS_ONSET: RecordKey = "cs_onset"  # Earliest CS onset in the trial (n_trials,)
DIMENSIONALITY: RecordKey = "dimensionality"  # Participation ratio of the feature trajectory during the CS (n_trials,)
WEIGHT_CHANGE: RecordKey = "weight_change"  # Frobenius norm of the change of the representation's weights (n_trials,)
FEATURES: RecordKey = "features"  # Feature vectors of the recorded trials (n_recorded, trial_duration, n_features)
FEATURE_TRIALS: RecordKey = "feature_trials"  # Indices of the trials whose features were recorded (n_recorded,)
