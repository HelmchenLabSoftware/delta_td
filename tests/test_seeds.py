"""Seed repeats of stochastic runs: file names, which runs are repeated, statistics over the saved runs."""

import numpy as np
import pytest

from deltatd.figures import helper
from deltatd.figures.ludvig2008 import f6_partial_reinforcement
from deltatd.figures.ludvig2012 import f2f3_acquisition
from deltatd.simulation import simulate
from deltatd.utils import constants as c
from deltatd.utils import ids, paths


def test_only_the_rnn_models_and_the_partial_schedule_are_stochastic():
    assert simulate.is_stochastic(ids.RNN) and simulate.is_stochastic(ids.RNN_PRESENCE_INIT_EXACT)
    assert not any(simulate.is_stochastic(m) for m in (ids.CSC, ids.MICROSTIMULUS, ids.PRESENCE, ids.DELTA))
    deterministic = f2f3_acquisition.CONDITIONS["isi25"]
    stochastic = f6_partial_reinforcement.CONDITIONS["p0.50"]
    assert not helper.accepts_seed(deterministic) and helper.accepts_seed(stochastic)
    assert helper.seeds_of(ids.CSC, deterministic) == (None,)
    assert helper.seeds_of(ids.RNN, deterministic) == c.SEEDS
    assert helper.seeds_of(ids.CSC, stochastic) == c.SEEDS


def test_seeded_schedule_differs_between_seeds():
    protocols = [f6_partial_reinforcement.CONDITIONS["p0.50"](seed=seed) for seed in c.SEEDS[:2]]
    assert [t.us_time for t in protocols[0]] != [t.us_time for t in protocols[1]]


def test_run_names():
    assert helper.run_name("csc", "isi25") == "csc_isi25"
    assert helper.run_name("rnn", "isi25", 3) == "rnn_isi25_seed3"


def test_jobs_are_queued_per_seed_and_run(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "SIMULATION_DIR", tmp_path)
    helper.JOBS = []
    try:
        helper.simulate_conditions("study", "experiment", {"isi25": f2f3_acquisition.CONDITIONS["isi25"]}, [ids.CSC, ids.RNN])
        jobs = helper.JOBS
    finally:
        helper.JOBS = None
    assert [(j.name, j.seed) for j in jobs] == [(ids.CSC, None)] + [(ids.RNN, seed) for seed in c.SEEDS]
    assert jobs[0].cost < jobs[1].cost
    job = helper.Job("study", "experiment", f2f3_acquisition.__name__, "isi25", ids.CSC, ids.CSC)
    job.run()
    assert job.path().exists() and job.path().name == "csc_isi25.npz"


def make_run(cr_level: np.ndarray, diverge_at: int | None = None) -> dict[str, np.ndarray]:
    n = len(cr_level)
    value = np.tile(cr_level[:, None], (1, 4)).astype(float)
    if diverge_at is not None:
        value[diverge_at] = 1e300
        value[diverge_at + 1 :] = np.nan
    return {
        ids.VALUE: value,
        ids.RESPONSE: value.copy(),
        ids.TD_ERROR: value.copy(),
        ids.CR_LEVEL: cr_level.astype(float),
        ids.PEAK_TIME: np.zeros(n),
        ids.DIMENSIONALITY: np.ones(n),
        ids.WEIGHT_CHANGE: np.zeros(n),
        ids.PROBE: np.zeros(n, dtype=bool),
        ids.LABEL: np.array(["train"] * n),
        ids.US_TIME: np.zeros(n, dtype=int),
        ids.CS_ONSET: np.zeros(n, dtype=int),
    }


def test_stats_average_over_seeds_and_mask_diverged_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "SIMULATION_DIR", tmp_path)
    runs = [make_run(np.array([1.0, 2.0, 3.0])), make_run(np.array([3.0, 4.0, 5.0])), make_run(np.array([1.0, 2.0, 3.0]), diverge_at=1)]
    for seed, run in zip(c.SEEDS, runs):
        simulate.save_results(helper.run_path("s", "e", ids.RNN, "cond", seed), run)
    stats = helper.load_stats("s", "e", ids.RNN, "cond")
    assert stats.n == 3 and stats.n_diverged == 1
    np.testing.assert_allclose(stats.mean[ids.CR_LEVEL], [5 / 3, 3.0, 4.0])  # the diverged run counts until it diverges
    assert stats.sd[ids.CR_LEVEL][1] == pytest.approx(1.0)
    assert np.isnan(stats.runs[2][ids.CR_LEVEL][1])  # masked from the divergence trial on
    mean, sd = stats.summary(lambda r: r[ids.CR_LEVEL][-1])
    assert (mean, sd) == (4.0, 1.0)  # NaN of the diverged run ignored
    assert "3 seeds" in stats.note and "1 diverged" in stats.note
    assert helper.load("s", "e", ids.RNN, "cond")[ids.CR_LEVEL][0] == 1.0


def test_deterministic_results_have_no_statistics(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "SIMULATION_DIR", tmp_path)
    simulate.save_results(helper.run_path("s", "e", ids.CSC, "cond"), make_run(np.array([1.0, 2.0])))
    stats = helper.load_stats("s", "e", ids.CSC, "cond")
    assert stats.n == 1 and stats.note == "" and not stats.sd[ids.CR_LEVEL].any()
    with pytest.raises(FileNotFoundError):
        helper.load_runs("s", "e", ids.PRESENCE, "cond")
