"""Run all simulations: the figure modules queue their runs, which are executed in parallel processes."""

from __future__ import annotations

import multiprocessing
import os

from deltatd.figures import helper, ludvig2008, ludvig2012, rnn
from deltatd.utils import ids

STUDIES = {ids.LUDVIG2012: ludvig2012, ids.LUDVIG2008: ludvig2008, ids.RNN_STUDY: rnn}


def collect_jobs(representations: list[str] = ids.ALL_REPRESENTATIONS, studies: list[str] = tuple(STUDIES)) -> list[helper.Job]:
    """The simulation runs of the requested studies and models, most expensive first."""
    helper.JOBS = []
    try:
        for study in studies:
            for module in STUDIES[study].MODULES:
                module.simulate(representations)
        jobs = helper.JOBS
    finally:
        helper.JOBS = None
    return sorted(jobs, key=lambda job: job.cost, reverse=True)


def run_job(job: helper.Job) -> str:
    job.run()
    return job.label


def run(
    representations: list[str] = ids.ALL_REPRESENTATIONS, studies: list[str] = tuple(STUDIES), workers: int | None = None
) -> None:
    """Run all simulations of all reproduced experiments (`workers` processes, default: all cores but one)."""
    jobs = collect_jobs(representations, studies)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"{len(jobs)} simulation runs on {workers} processes")
    if workers == 1:
        for k, job in enumerate(jobs, 1):
            print(f"  [{k}/{len(jobs)}] {job.label}", flush=True)
            job.run()
    else:
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            for k, label in enumerate(pool.imap_unordered(run_job, jobs), 1):
                print(f"  [{k}/{len(jobs)}] {label}", flush=True)
    print("All simulations complete!")
