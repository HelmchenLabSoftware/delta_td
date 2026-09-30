# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this repo is

Simulation code comparing four models of real-time US prediction in classical conditioning: the presence, complete
serial compound and microstimulus representations with TD(lambda) from Ludvig, Sutton & Kehoe (2012) and the
integrated value-change ("delta-TD") learner from Schoenfeld et al. (2024, bioRxiv 10.1101/2021.12.28.474360v3,
Supplementary Note 2). Architecture: string identifiers in `deltatd/utils/ids.py`, all parameters in
`deltatd/utils/constants.py`, one figure module per manuscript figure with a `simulate()` and a `plot()` function,
results in `results/` (gitignored, regenerated in a minute), figures in `figures/` (committed). The layout was inspired
by `unibe-cns/TopDownOFC` but should evolve with this project's needs rather than mirror it.

## Commands

```
python main.py simulate   # run all simulations -> results/simulation/<study>/<experiment>/*.npz (~1 min)
python main.py plot       # plot all figures -> figures/<study>/*.pdf and *.png (default command)
python main.py all        # both; --representations csc microstimulus presence delta and --studies ludvig2012 ludvig2008 restrict
python -m pytest          # unit tests (tests/)
```

Environment: `.venv` created with `uv venv .venv --python 3.12` and `uv pip install -e ".[dev]"`; numpy and
matplotlib only (no torch). Figures are saved as PDF and PNG, look at the PNGs to check a plot.

## Architecture

- `simulation/representations.py`: `Representation.step(present, onset) -> x`. Stimuli are indexed in the order of
  `ids.STIMULI = ("A", "B", "US")`; the US is a stimulus too (it spawns microstimuli).
- `simulation/learners.py`: `Learner.step(x, reward, us_present, trial_end) -> StepOutput(value, td_error)`.
  `TDLambda` keeps state across trials; `DeltaTD` resets its integrated value and eligibility trace at trial start.
- `simulation/response.py`: CR = thresholded leaky integrator of the value, restarted at every trial.
- `simulation/tasks.py`: `Trial` dataclass and protocol builders. CSs coterminate with the US; the earliest CS onset is
  `CS_ONSET_TIME`. Probe trials are unreinforced and labelled (`A_alone`, `B_alone`, `compound`, `probe`).
- `simulation/simulate.py`: `build_model(representation_id)` assembles representation + learner + response;
  `run_protocol(model, protocol)` returns a dict of arrays keyed by `ids.RecordKey`, saved with `np.savez_compressed`.
- `figures/<study>/<figure>.py`: `CONDITIONS` maps condition name -> protocol factory; `simulate()` runs every
  representation through every condition via `helper.simulate_conditions`; `plot()` loads with `helper.load`.
  The 2008 modules pass `model_factory=simulate.build_dopamine_model` (parameters in `simulate.DOPAMINE_PARAMETERS`)
  and use only `constants.DA_REPRESENTATIONS`. `fs1_delta_variants` simulates the delta-TD consumption rules itself.

Adding an experiment: add an `ExperimentID` in `ids.py`, parameters in `constants.py`, a protocol builder in
`tasks.py`, a figure module registered in `figures/<study>/__init__.py::MODULES`, and a test in `tests/`.
Adding a model: a `Representation` and/or `Learner`, a `RepresentationID`, a branch in `build_model`, a colour in
`figures/helper.py`.

## Conventions

- Time is in time steps (300 per trial); parameters follow the Appendix of Ludvig et al. (2012) unless stated.
- Models are deterministic; there is no seed plumbing yet.
- Delta-TD counts as its own stimulus representation (brief onset pulses + integrated value changes), combined with
  its own TD variant; classic TD(lambda) is not compatible with it. Applying the delta learning rule to the other
  representations is a possible later experiment, not a goal.
- Open modelling questions about the delta-TD model (reset rule, discounting, input events) are listed in
  `docs/model_notes.md`; do not silently change `DELTA_*` defaults, flag them.
