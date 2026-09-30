"""Parameter values.

The classical conditioning parameters are those of the Appendix of Ludvig, Sutton & Kehoe (2012), which were used
unchanged for all their simulations. The delta-TD parameters follow Supplementary Table 1 of Schoenfeld et al. (2024)
where applicable; see docs/model_notes.md for the choices that had to be made to transfer the model to Pavlovian tasks.
"""

# --- Ludvig et al. (2012), Appendix -----------------------------------------------------------------------------
# Learning rule
DISCOUNT: float = 0.97  # Discount factor gamma
TRACE_DECAY: float = 0.95  # Eligibility trace decay rate lambda
STEP_SIZE: float = 0.05  # Learning rate alpha
# Response model (thresholded leaky integrator)
RESPONSE_THRESHOLD: float = 0.25  # theta
RESPONSE_DECAY: float = 0.9  # nu
# Stimulus representations
MEMORY_DECAY: float = 0.985  # Memory trace decay constant d of the microstimulus representation
N_MICROSTIMULI: int = 6  # Number of microstimuli m per stimulus
MICROSTIMULUS_WIDTH: float = 0.08  # Width sigma of the Gaussian basis functions
PRESENCE_SALIENCE: float = 0.2  # Feature value x of the presence element while a stimulus is on
# Protocol
US_MAGNITUDE: float = 1.0  # US intensity, lasting a single time step
TRIAL_DURATION: int = 300  # Time steps per trial
CS_INCLUDES_US_STEP: bool = False  # CS duration = ISI (CS ends when the US arrives); True keeps it on during the US step
CS_ONSET_TIME: int = 20  # Time step of the earliest CS onset within a trial (baseline before the CS)

# Ludvig et al. (2012) experiments
N_TRIALS_ACQUISITION: int = 200  # Acquisition set (Fig. 2 and 3)
ACQUISITION_ISIS: tuple[int, ...] = (0, 5, 10, 25, 50, 100)
ACQUISITION_EXAMPLE_ISI: int = 25  # ISI used for the US prediction time courses of Fig. 2
ACQUISITION_EXAMPLE_TRIALS: tuple[int, ...] = (1, 10, 50, 200)  # Trials shown in Fig. 2 (1-based)
N_TRIALS_TIMING: int = 500  # Timing set (Fig. 4)
TIMING_ISIS: tuple[int, ...] = (10, 25, 50, 100)
PROBE_EVERY: int = 5  # Every 5th trial of the timing set is an unreinforced probe trial with the CS on for 2 x ISI
N_TRIALS_BLOCKING_PHASE: int = 200  # Trials per phase of the blocking experiments (Fig. 5 and 6)
BLOCKING_CONDITIONS: dict[str, tuple[int, int, int]] = {  # name -> (ISI of A in phase 1, ISI of A in phase 2, ISI of B)
    "identical": (50, 50, 50),  # Fig. 5a
    "b_later": (50, 50, 25),  # Fig. 5b
    "b_earlier": (25, 25, 50),  # Fig. 5c
    "isi_change": (100, 25, 25),  # Fig. 6
}
N_TRIALS_OVERSHADOWING: int = 200  # Overshadowing (Fig. 7)
OVERSHADOWING_ISI_B: int = 25  # ISI of the overshadowed stimulus B
OVERSHADOWING_CONDITIONS: dict[str, int | None] = {"same": 25, "long": 50, "longer": 100, "none": None}  # ISI of A
OVERSHADOWING_EXAMPLE: str = "long"  # Condition whose time courses are shown in Fig. 7c

# --- Delta-TD model (Schoenfeld et al. 2024, Supplementary Note 2) ------------------------------------------------
DELTA_DISCOUNT: float = DISCOUNT  # Same discounting as the other models (the paper used 1); V_hat grows by 1/gamma per step
DELTA_TRACE_DECAY: float = 1.0  # The paper integrated the eligibility trace without decay within a trial
DELTA_STEP_SIZE: float = STEP_SIZE  # Paper: 0.016 with 18-step trials; set equal to the other models for comparability
# Fair-comparison settings: Ludvig's models get the reward timing from the US being a stimulus (its microstimuli
# learn negative weights) and from the CS terminating at the US. Delta-TD therefore receives the same events as
# learnable inputs instead of a hard-coded reward-timing reset (see docs/model_notes.md).
DELTA_RESET_RULE: str = "offset"  # CS termination consumes the prediction (see ids.ResetRule and docs/model_notes.md)
DELTA_US_AS_STIMULUS: bool = True  # The US onset is an input event whose weight learns to cancel the prediction
DELTA_INCLUDE_OFFSETS: bool = False  # Whether CS offsets are input events too (the CS offset coincides with the US)
DELTA_DECAY: float = 1.0  # Leak of the integrated value estimate per step (0.985 would match the MS memory decay)
