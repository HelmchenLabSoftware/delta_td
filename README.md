# delta_td

## Description

Simulations comparing the integrated value-change model of temporal-difference (TD) learning used in
'[Unsigned temporal difference errors in cortical L5 dendrites during learning](https://www.biorxiv.org/content/10.1101/2021.12.28.474360v3)'
(Schoenfeld et al., Supplementary Note 2; here called the $\Delta$-TD model) with the classical temporal stimulus
representations of the TD model of conditioning:

* the presence representation, the complete serial compound (CSC) and the microstimulus (MS) representation
  evaluated in [Ludvig, Sutton & Kehoe (2012), *Learning & Behavior*](https://doi.org/10.3758/s13420-012-0082-6);
* the microstimulus model of dopamine responses of
  [Ludvig, Sutton & Kehoe (2008), *Neural Computation*](https://doi.org/10.1162/neco.2008.11-07-654).

The simulations reproduce the classical conditioning experiments of Ludvig et al. (2012) (acquisition, ISI effects,
response timing, blocking, overshadowing) and the dopamine / TD-error experiments of Ludvig et al. (2008) (simple
acquisition, reward omission, partial reinforcement, early reward, multiple cues) and add the $\Delta$-TD model to
each comparison. See `docs/model_notes.md` for the equations, the modelling choices and the open questions.

### Installation

With [uv](https://docs.astral.sh/uv/):
```
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e ".[dev]"
```
or with conda:
```
conda env create -f environment.yml
conda activate deltatd
```

### Execution

Run all simulations (about one minute; results are not versioned) and plot all figures:
```
python main.py all
python main.py simulate       # simulations only -> results/simulation
python main.py plot           # figures only (needs existing results) -> figures/
python main.py all --representations csc delta   # restrict to some models
python main.py all --studies ludvig2008          # restrict to one paper
```
Run the tests:
```
python -m pytest
```

### Layout

```
main.py                         command line dispatcher
deltatd/
  run_all.py, plot_all.py       run every simulation / plot every figure
  utils/                        paths, string identifiers (ids.py) and all parameter values (constants.py)
  simulation/
    representations.py          presence, CSC, microstimulus and onset representations
    learners.py                 TD(lambda) (Ludvig) and DeltaTD (Schoenfeld) learning rules
    response.py                 thresholded leaky-integrator response rule
    tasks.py                    trial and protocol builders (acquisition, timing, blocking, overshadowing)
    simulate.py                 model assembly, protocol runner, result I/O
  figures/
    helper.py                   plotting helpers and the simulate-all-conditions loop
    ludvig2012/                 one module per figure of Ludvig et al. (2012) + fs1_delta_variants, each with simulate() and plot()
    ludvig2008/                 one module per figure of Ludvig et al. (2008), dopamine / TD-error observables
results/simulation/<study>/<experiment>/<representation>_<condition>.npz   (gitignored)
figures/<study>/<figure>.pdf|png
docs/model_notes.md             equations, notation mapping, open modelling questions, reproduction status
tests/                          pytest unit tests of the representations, protocols and learning outcomes
```
