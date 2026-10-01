"""Parameter values.

The classical conditioning parameters are those of the Appendix of Ludvig, Sutton & Kehoe (2012), which were used
unchanged for all their simulations. The delta-TD parameters follow Supplementary Table 1 of Schoenfeld et al. (2024)
where applicable; see docs/model_notes.md for the choices that had to be made to transfer the model to Pavlovian tasks.
"""

# --- Reproducibility --------------------------------------------------------------------------------------------
# The models are deterministic; the only random elements are the reward schedule of the partial reinforcement
# experiment and the initial weights of the recurrent network. Both draw from their own `np.random.default_rng`
# seeded from SEED, so every run is bit-for-bit reproducible (tests/test_reproducibility.py); the stochastic runs are
# repeated over SEEDS and the figures show statistics over the seeds.
SEED: int = 0  # Master seed of all random elements
N_SEEDS: int = 5  # Seeds over which every stochastic run (RNN models, partial reinforcement schedule) is repeated
SEEDS: tuple[int, ...] = tuple(range(SEED, SEED + N_SEEDS))  # Figures show mean and SD over these seeds

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

# --- Ludvig et al. (2008), Section 2 --------------------------------------------------------------------------------
# Same model structure as 2012 but with the dopamine-experiment parameters; the observable is the TD error, not a CR.
DA_DISCOUNT: float = 0.98
DA_TRACE_DECAY: float = 0.95
DA_STEP_SIZE: float = 0.01
DA_N_MICROSTIMULI: int = 50
DA_MICROSTIMULUS_WIDTH: float = 0.08
DA_MEMORY_DECAY: float = 0.985
DA_STEPS_PER_SECOND: int = 20
DA_TRIAL_DURATION: int = 500  # Intertrial interval of 500 time steps between trial onsets
DA_CSC_MAX_DURATION: int = 200  # Length of the CSC delay line (10 s): covers the trial events, shorter than the ITI so
# that the reward's delay line cannot predict the next trial's cue (the paper does not state this length)
DA_CUE_TIME: int = 20  # Cue onset within the trial (baseline before the cue)
DA_REWARD_DELAY: int = 20  # Reward 1 s after the cue
DA_CUE_LASTS_UNTIL_REWARD: bool = True  # Cue stays on until the usual reward time (delay conditioning); False: 1 step
DA_N_TRIALS: int = 1000  # Simple acquisition, omission, early reward, multiple cues
DA_EXAMPLE_TRIALS: tuple[int, ...] = (1, 100, 1000)  # Trials shown in Fig. 3 (1-based)
DA_N_TRIALS_PARTIAL: int = 500  # Partial reinforcement
DA_REWARD_PROBABILITIES: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
DA_PARTIAL_SEED: int = SEED  # Seed of the reward schedule of the partial reinforcement experiment
DA_N_EARLY_PROBES: int = 15  # Early reward probe trials after training
DA_EARLY_REWARD_DELAY: int = 10  # Reward 0.5 s after the cue on probe trials
DA_SECOND_CUE_DELAY: int = 40  # Second cue 2 s after the first cue (multiple cues)
DA_MULTI_REWARD_DELAY: int = 60  # Reward 3 s after the first cue (multiple cues)
DA_MULTI_EXAMPLE_TRIALS: tuple[int, ...] = (50, 1000)  # Early and late training (Fig. 8)
DA_REPRESENTATIONS: tuple[str, ...] = ("csc", "microstimulus", "delta", "rnn")  # Presence was not used in 2008

# --- Delta-TD variants (supplementary figure) --------------------------------------------------------------------
DELTA_VARIANT_RULES: tuple[str, ...] = ("offset", "event", "us", "trial_end")
DELTA_VARIANT_ISI: int = 25  # ISI of the acquisition time courses
DELTA_VARIANT_PROBE_ISI: int = 50  # ISI of the timing-set probe trials

# --- Delta-TD model (Schoenfeld et al. 2024, Supplementary Note 2) ------------------------------------------------
DELTA_DISCOUNT: float = DISCOUNT  # Same discounting as the other models (the paper used 1); V_hat grows by 1/gamma per step
DELTA_TRACE_DECAY: float = 1.0  # The paper integrated the eligibility trace without decay within a trial
DELTA_STEP_SIZE: float = STEP_SIZE  # Paper: 0.016 with 18-step trials; set equal to the other models for comparability
# Fair-comparison settings: Ludvig's models get the reward timing from the US being a stimulus (its microstimuli
# learn negative weights) and from the CS terminating at the US. Delta-TD therefore receives the same events as
# learnable inputs instead of a hard-coded reward-timing reset (see docs/model_notes.md).
# The integrated prediction is cashed in (R_hat = V_hat, V_hat restarts) when the US arrives, or, if no US comes,
# when the CS terminates. The CS offset is not an input; it only bounds the prediction on unreinforced trials.
DELTA_RESET_RULE: str = "event"  # See ids.ResetRule and docs/model_notes.md
DELTA_US_AS_STIMULUS: bool = True  # The US onset is an input event whose weight learns to cancel the prediction
DELTA_INCLUDE_OFFSETS: bool = False  # Whether CS offsets are input events too (the CS offset coincides with the US)
DELTA_DECAY: float = 1.0  # Leak of the integrated value estimate per step (0.985 would match the MS memory decay)

# --- Learnable recurrent representation (RNN + TD(lambda) readout) --------------------------------------------------
# h_t = [W h_{t-1} + U onset_t]_+ with RNN_UNITS_PER_STIMULUS units nominally assigned to every stimulus (the plastic
# recurrent weights connect all units). See docs/model_notes.md for the plasticity rule and the initializations.
RNN_UNITS_PER_STIMULUS: int = 100  # Longest ISI of the experiments: the delay-line (CSC) init needs a unit per step
RNN_INIT: str = "random"  # Default initialization of the plain "rnn" model (see ids.RNN_INITS)
RNN_STEP_RATIO: float = 2e-2  # Recurrent step size / readout step size of the local credit assignment: fastest ratio
# at which every initialization is stable and the CR level has settled (relative SD < 1% over the last 20 trials)
RNN_EXACT_STEP_RATIO: float = 5e-4  # ... of the exact credit assignment (same criterion; the microstimulus init
# diverges with the exact rule at any ratio above 1e-6, see docs/model_notes.md)
RNN_STEP_RATIOS: tuple[float, ...] = (0.0, 1e-6, 1e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3, 1e-2, 2e-2, 3e-2, 5e-2, 1e-1)
RNN_CONVERGENCE_TRIALS: int = 20  # Trials at the end of a run over which convergence of the CR level is judged
RNN_INIT_NOISE: float = 1e-4  # Std of the Gaussian noise added to the structured initial weights (1e-3 destabilizes
# the ill-conditioned microstimulus block: spectral radius 1.06 instead of 0.98)
RNN_CREDIT: str = "local"  # Credit assignment to recurrent synapses: "local" eligibility traces or "exact" gradient
RNN_CREDIT_HORIZON: int = 150  # Time steps over which the exact gradient is propagated backwards
RNN_RANDOM_SPECTRAL_RADIUS: float = 1.4  # Random init: rectification halves the gain, so activity persists ~100 steps
RNN_SEED: int = SEED  # Seed of the random initial weights and of the initialization noise
RNN_PRESENCE_INHIBITION: float = 1.0  # Presence init: US onset weight (negative) that switches the CS units off
RNN_MS_EMBEDDING_ORDER: int = 4  # Microstimulus init: linear system fitted to the delay embedding of the microstimuli
RNN_MS_HIDDEN_SCALE: float = 0.1  # Microstimulus init: amplitude of the auxiliary (delayed copy) units
RNN_MS_RIDGE: float = 1e-2  # Microstimulus init: ridge penalty of the autoregression fit. The lagged microstimuli are
# nearly collinear, so the unregularized fit (1e-6) has large cancelling coefficients that make the exact credit
# assignment diverge; 1e-2 is the smallest value stable on all seeds, at the price of a 15% reproduction error
# (docs/model_notes.md, diagnosis of 2026-10-01)
RNN_EXAMPLE_ISI: int = ACQUISITION_EXAMPLE_ISI  # ISI of the initialization and step-size analyses
RNN_EXAMPLE_TRIALS: tuple[int, ...] = ACQUISITION_EXAMPLE_TRIALS  # Trials whose features are recorded (1-based)
