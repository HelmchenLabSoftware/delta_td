# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this repo is

Simulation code comparing five models of real-time US prediction in classical conditioning: the presence, complete
serial compound and microstimulus representations with TD(lambda) from Ludvig, Sutton & Kehoe (2012), the
integrated value-change ("delta-TD") learner from Schoenfeld et al. (2024, bioRxiv 10.1101/2021.12.28.474360v3,
Supplementary Note 2) and a learnable representation (rectified recurrent network driven by onset pulses, plastic
recurrent weights, TD(lambda) readout; `rnn`). Architecture: string identifiers in `deltatd/utils/ids.py`, all parameters in
`deltatd/utils/constants.py`, one figure module per manuscript figure with a `simulate()` and a `plot()` function,
results in `results/` (gitignored, regenerated in a minute), figures in `figures/` (committed). The layout was inspired
by `unibe-cns/TopDownOFC` but should evolve with this project's needs rather than mirror it.

## Commands

```
python main.py simulate   # run all simulations -> results/simulation/<study>/<experiment>/*.npz (parallel, --workers N)
python main.py plot       # plot all figures -> figures/<study>/*.pdf and *.png (default command)
python main.py all        # both; --representations csc microstimulus presence delta rnn and --studies ludvig2012 ludvig2008 rnn restrict
python -m pytest          # unit tests (tests/)
```

Environment: `.venv` created with `uv venv .venv --python 3.12` and `uv pip install -e ".[dev]"`; numpy and
matplotlib only (no torch). Figures are saved as PDF and PNG, look at the PNGs to check a plot.

## Architecture

- `simulation/representations.py`: `Representation.step(present, onset) -> x`. Stimuli are indexed in the order of
  `ids.STIMULI = ("A", "B", "US")`; the US is a stimulus too (it spawns microstimuli). `RecurrentNetwork` is the
  learnable representation: `learn(td_error, readout)` (called by `run_protocol` after the learner's step) applies
  the three-factor rule to the recurrent weights; `initial_weights` builds the random init or the blocks that
  reproduce the CSC, microstimulus or presence representation (model variants `rnn_csc`, `rnn_microstimulus`,
  `rnn_presence` in `ids.MODEL_VARIANTS`; `*_exact` variants use the exact backward-propagated credit assignment
  instead of the local eligibility, with its own default step ratio `RNN_EXACT_STEP_RATIO`).
- `simulation/learners.py`: `Learner.step(x, reward, us_present, trial_end) -> StepOutput(value, td_error)`.
  `TDLambda` keeps state across trials; `DeltaTD` resets its integrated value and eligibility trace at trial start.
- `simulation/response.py`: CR = thresholded leaky integrator of the value, restarted at every trial.
- `simulation/tasks.py`: `Trial` dataclass and protocol builders. CSs coterminate with the US; the earliest CS onset is
  `CS_ONSET_TIME`. Probe trials are unreinforced and labelled (`A_alone`, `B_alone`, `compound`, `probe`).
- `simulation/simulate.py`: `build_model(model_id)` assembles representation + learner + response (model variants
  apply their overrides); `run_protocol(model, protocol)` returns a dict of arrays keyed by `ids.RecordKey`, saved
  with `np.savez_compressed` (`record_features_at` stores feature vectors of chosen trials, `feature_metrics` the
  dimensionality of the CS state per trial).
- `figures/<study>/<figure>.py`: `CONDITIONS` maps condition name -> protocol factory; `simulate()` runs every
  representation through every condition via `helper.simulate_conditions`; `plot()` loads with `helper.load`.
  The 2008 modules pass `model_factory=simulate.build_dopamine_model` (parameters in `simulate.DOPAMINE_PARAMETERS`)
  and use only `constants.DA_REPRESENTATIONS`. `fs1_delta_variants` simulates the delta-TD consumption rules itself.
  `figures/rnn/` compares the RNN initializations and scans the recurrent step size (study `rnn`, only runs when
  `rnn` is among the requested models). The RNN runs take about 0.3 ms per time step (roughly 30 min for everything
  on one core and seed; the full multi-seed set is about 17 CPU hours, run in parallel processes by `run_all`).

Adding an experiment: add an `ExperimentID` in `ids.py`, parameters in `constants.py`, a protocol builder in
`tasks.py`, a figure module registered in `figures/<study>/__init__.py::MODULES`, and a test in `tests/`.
Adding a model: a `Representation` and/or `Learner`, a `RepresentationID`, a branch in `build_model`, a colour in
`figures/helper.py`.

## Conventions

- Time is in time steps (300 per trial); parameters follow the Appendix of Ludvig et al. (2012) unless stated.
- Models are deterministic. The only random elements (partial reinforcement schedule, RNN initial weights) use
  local `np.random.default_rng` generators seeded from `constants.SEED`; never use the global numpy RNG
  (`tests/test_reproducibility.py` checks that two runs are identical). Stochastic runs (any RNN model, any protocol
  factory with a `seed` argument) are repeated over `constants.SEEDS` and saved as `<model>_<condition>_seed<k>.npz`;
  deterministic runs are saved once without suffix. Figures load `helper.load_stats` (mean/SD over seeds, diverged
  runs masked from their divergence trial) and draw means with SD bands or error bars (`helper.plot_band`,
  `helper.errorbars`, `(mean, sd)` values in `grouped_bars`); `helper.load` gives the first seed as an example.
- Simulations are queued as `helper.Job`s (a figure module's `CONDITIONS[name]` factory + model + seed) and run in
  parallel by `run_all` (`main.py simulate --workers N`); `helper.submit` runs a job inline when no queue is open.
- Delta-TD counts as its own stimulus representation (brief onset pulses + integrated value changes), combined with
  its own TD variant; classic TD(lambda) is not compatible with it. Applying the delta learning rule to the other
  representations is a possible later experiment, not a goal.
- Open modelling questions about the delta-TD model (reset rule, discounting, input events) are listed in
  `docs/model_notes.md`; do not silently change `DELTA_*` defaults, flag them.
- The RNN model receives onset pulses only (no presence input, no offset pulse); its `RNN_*` defaults (units, step
  ratio, credit assignment, init noise, spectral radius) are decisions recorded in `docs/model_notes.md`, flag changes.
