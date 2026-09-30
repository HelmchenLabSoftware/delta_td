"""Supplementary figure: the delta-TD model under the different rules for consuming the integrated prediction.

Rows: consumption rules (`DELTA_VARIANT_RULES`). Columns: (a) US prediction during acquisition (ISI 25, trial 200),
(b) response on the last timing-set probe trial (ISI 50, CS extended to 100 steps), (c) CR levels on the probe trials
of blocking with an earlier CSB. The "trial end only" variant uses gamma = 1, the others the default discount.
"""

from __future__ import annotations

import numpy as np

from deltatd.figures import helper
from deltatd.simulation import simulate as sim
from deltatd.simulation import tasks
from deltatd.utils import constants as c
from deltatd.utils import ids

STUDY = ids.LUDVIG2012
EXPERIMENT = ids.DELTA_VARIANTS
TASKS = {
    "acquisition": lambda: tasks.acquisition_protocol(c.DELTA_VARIANT_ISI),
    "timing": lambda: tasks.timing_protocol(c.DELTA_VARIANT_PROBE_ISI),
    "blocking_b_earlier": lambda: tasks.blocking_protocol(*c.BLOCKING_CONDITIONS["b_earlier"]),
}


def variant_model(rule: str) -> sim.Model:
    gamma = 1.0 if rule == ids.RESET_AT_TRIAL_END else c.DELTA_DISCOUNT
    return sim.build_model(ids.DELTA, learner_kwargs={"reset_rule": rule, "gamma": gamma})


def simulate(representations: list[str] = ids.ALL_REPRESENTATIONS) -> None:
    if ids.DELTA not in representations:
        return
    for rule in c.DELTA_VARIANT_RULES:
        for task, make_protocol in TASKS.items():
            print(f"  {EXPERIMENT}: {rule} / {task}", flush=True)
            results = sim.run_protocol(variant_model(rule), make_protocol())
            sim.save_results(helper.run_path(STUDY, EXPERIMENT, rule, task), results)
    print(f"  {EXPERIMENT}: done")



def plot(representations: list[str] = ids.ALL_REPRESENTATIONS) -> None:
    if ids.DELTA not in representations:
        return
    helper.set_style()
    rules = list(c.DELTA_VARIANT_RULES)
    fig, axes = helper.representation_panels(
        ["acquisition", "timing", "blocking"],
        n_rows=len(rules),
        titles=[
            f"US prediction, acquisition ISI {c.DELTA_VARIANT_ISI}",
            f"Response on probe trial, ISI {c.DELTA_VARIANT_PROBE_ISI}",
            "Blocking, CSB earlier",
        ],
        sharey=False,
    )
    for row, rule in enumerate(rules):
        color = helper.COLORS[ids.DELTA]
        acquisition = sim.load_results(helper.run_path(STUDY, EXPERIMENT, rule, "acquisition"))
        for k, trial in enumerate(c.ACQUISITION_EXAMPLE_TRIALS):
            shade = 0.3 + 0.7 * k / max(len(c.ACQUISITION_EXAMPLE_TRIALS) - 1, 1)
            axes[row, 0].plot(
                helper.relative_time(acquisition, trial - 1), acquisition[ids.VALUE][trial - 1], color=color, alpha=shade, label=f"Trial {trial}"
            )
        axes[row, 0].axvline(c.DELTA_VARIANT_ISI, color="k", ls=":", lw=0.8)
        axes[row, 0].set_xlim(-10, 2 * c.DELTA_VARIANT_ISI + 10)
        axes[row, 0].set_ylabel(f"{ids.DELTA_VARIANT_LABELS[rule]}\n\nUS prediction")

        timing = sim.load_results(helper.run_path(STUDY, EXPERIMENT, rule, "timing"))
        probe = np.flatnonzero(timing[ids.PROBE])[-1]
        axes[row, 1].plot(helper.relative_time(timing, probe), timing[ids.RESPONSE][probe], color=color)
        axes[row, 1].axvline(c.DELTA_VARIANT_PROBE_ISI, color="k", ls=":", lw=0.8)
        axes[row, 1].axvline(2 * c.DELTA_VARIANT_PROBE_ISI, color="k", ls="--", lw=0.8)
        axes[row, 1].set_xlim(-10, c.TRIAL_DURATION - c.CS_ONSET_TIME)
        axes[row, 1].set_ylabel("CR level")

        blocking = sim.load_results(helper.run_path(STUDY, EXPERIMENT, rule, "blocking_b_earlier"))
        levels = {label: float(blocking[ids.CR_LEVEL][helper.probe_index(blocking, label)]) for label in helper.PROBE_LABELS}
        axes[row, 2].bar(range(len(levels)), list(levels.values()), color=[f"C{k}" for k in range(len(levels))])
        axes[row, 2].set_xticks(range(len(levels)))
        axes[row, 2].set_xticklabels([helper.PROBE_LABELS[label] for label in levels], rotation=15)
        axes[row, 2].set_ylabel("CR level")
    axes[0, 0].legend(frameon=False)
    axes[-1, 0].set_xlabel("Time steps from CS onset")
    axes[-1, 1].set_xlabel("Time steps from CS onset")
    helper.save_figure(fig, STUDY, "fs1_delta_variants")
