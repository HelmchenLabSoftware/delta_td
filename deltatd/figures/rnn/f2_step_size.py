"""Scan of the recurrent step size (relative to the readout step size), acquisition at ISI 25.

For each credit assignment rule (local, exact), initialization and seed, the acquisition set of ISI 25 is run with
every ratio of `RNN_STEP_RATIOS`. One figure per credit rule shows the CR level and the dimensionality of the CS
state over trials and the US prediction on the last trial (rows, means over seeds) per initialization (columns) and
ratio (lines). A summary figure shows, against the ratio, the final CR level, the trials needed to reach 90% of it
and its fluctuation over the last trials (mean and SD over seeds), which are the criteria for the default ratios
(`RNN_STEP_RATIO`, `RNN_EXACT_STEP_RATIO`).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from deltatd.figures import helper
from deltatd.simulation import tasks
from deltatd.utils import constants as c
from deltatd.utils import ids

STUDY = ids.RNN_STUDY
EXPERIMENT = ids.RNN_STEP_SIZE
DEFAULTS = {ids.RNN_LOCAL_CREDIT: c.RNN_STEP_RATIO, ids.RNN_EXACT_CREDIT: c.RNN_EXACT_STEP_RATIO}


def condition(ratio: float) -> str:
    return f"ratio{ratio:g}"


CONDITIONS = {condition(ratio): (lambda: tasks.acquisition_protocol(c.RNN_EXAMPLE_ISI)) for ratio in c.RNN_STEP_RATIOS}


def simulate(representations: list[str] = ids.ALL_REPRESENTATIONS) -> None:
    if ids.RNN not in representations:
        return
    for models in ids.RNN_MODELS_BY_CREDIT.values():
        for model_id in models.values():
            for ratio in c.RNN_STEP_RATIOS:
                for seed in c.SEEDS:
                    helper.submit(
                        helper.Job(
                            STUDY, EXPERIMENT, __name__, condition(ratio), model_id, model_id, seed,
                            representation_kwargs={"step_ratio": ratio}, run_kwargs={"feature_metrics": True},
                        )
                    )


def plot(representations: list[str] = ids.ALL_REPRESENTATIONS) -> None:
    if ids.RNN not in representations:
        return
    helper.set_style()
    for credit, models in ids.RNN_MODELS_BY_CREDIT.items():
        plot_scan(credit, models)
    plot_summary()


def plot_scan(credit: str, models: dict[str, str]) -> None:
    """Rows: CR level, dimensionality, last-trial prediction. Columns: initializations. Lines: ratios."""
    fig, axes = helper.representation_panels(
        list(models.values()), n_rows=3, sharey="row", titles=[f"{ids.RNN_INIT_LABELS[init]}, {ids.RNN_CREDIT_LABELS[credit]}" for init in models]
    )
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(c.RNN_STEP_RATIOS)))
    isi = c.RNN_EXAMPLE_ISI
    cr_max = 0.0
    n_seeds = 1
    for col, model_id in enumerate(models.values()):
        for color, ratio in zip(colors, c.RNN_STEP_RATIOS):
            stats = helper.load_stats(STUDY, EXPERIMENT, model_id, condition(ratio))
            n_seeds = max(n_seeds, stats.n)
            trials = np.arange(1, len(stats[ids.CR_LEVEL]) + 1)
            label = f"ratio {ratio:g}" + (" (frozen)" if ratio == 0 else "") + (" (default)" if ratio == DEFAULTS[credit] else "")
            if np.any(np.isfinite(stats[ids.CR_LEVEL])):
                cr_max = max(cr_max, float(np.nanmax(stats[ids.CR_LEVEL])))
            helper.plot_band(axes[0, col], trials, stats, ids.CR_LEVEL, color=color, label=label if col == 0 else None)
            helper.plot_band(axes[1, col], trials, stats, ids.DIMENSIONALITY, color=color)
            helper.plot_band(axes[2, col], helper.relative_time(stats, len(trials) - 1), stats, ids.VALUE, len(trials) - 1, color=color)
            for run in stats.runs:  # mark where each diverged seed died
                if (diverged := helper.divergence_trial(run)) is not None:
                    axes[0, col].plot(diverged, 0, "x", color=color, ms=5, clip_on=False)
        axes[2, col].axvline(isi, color="k", ls=":", lw=0.8)
        axes[2, col].set_xlim(-10, 2 * isi + 10)
        axes[0, col].set_xlabel("Trials")
        axes[1, col].set_xlabel("Trials")
        axes[2, col].set_xlabel("Time steps from CS onset")
    note = f"\nmean and SD band over {n_seeds} seeds" if n_seeds > 1 else ""
    fig.legend(loc="outside right upper", frameon=False, title=f"x: a seed diverged on that trial{note}")
    axes[0, 0].set_ylabel("CR level")
    axes[0, 0].set_ylim(0, 1.1 * cr_max)  # a diverged run would otherwise blow up the axis
    axes[1, 0].set_ylabel("Participation ratio (CS state)")
    axes[1, 0].set_yscale("log")
    axes[2, 0].set_ylabel(f"US prediction, trial {c.N_TRIALS_ACQUISITION}")
    helper.save_figure(fig, STUDY, f"f3_rnn_step_size_{credit}")


def final_cr(results: dict[str, np.ndarray]) -> float:
    cr = results[ids.CR_LEVEL]
    return float(cr[-c.RNN_CONVERGENCE_TRIALS :].mean()) if np.all(np.isfinite(cr)) else np.nan


def plot_summary() -> None:
    """Final CR level, trials to 90% of it and its late fluctuation against the ratio; columns: initializations."""
    inits = list(ids.RNN_INITS)
    fig, axes = helper.representation_panels(
        inits, n_rows=3, sharey="row", titles=[ids.RNN_INIT_LABELS[init] for init in inits]
    )
    ratios = np.array(c.RNN_STEP_RATIOS)
    x = np.where(ratios > 0, ratios, ratios[ratios > 0].min() / 3)  # the frozen network is drawn left of the axis
    n_seeds = 1
    for credit, models in ids.RNN_MODELS_BY_CREDIT.items():
        ls = helper.LINESTYLES[credit]
        for col, init in enumerate(inits):
            model_id = models[init]
            color = helper.COLORS[model_id]
            final, speed, fluctuation, diverged = [], [], [], []
            for ratio in c.RNN_STEP_RATIOS:
                stats = helper.load_stats(STUDY, EXPERIMENT, model_id, condition(ratio))
                n_seeds = max(n_seeds, stats.n)
                final.append(stats.summary(final_cr))
                speed.append(stats.summary(lambda r: helper.trials_to_converge(r[ids.CR_LEVEL])))
                fluctuation.append(stats.summary(lambda r: helper.final_fluctuation(r[ids.CR_LEVEL])))
                diverged.append(stats.n_diverged > 0)
            label = ids.RNN_CREDIT_LABELS[credit]
            helper.errorbars(axes[0, col], x, final, fmt="o" + ls, color=color, ms=3, label=label)
            helper.errorbars(axes[1, col], x, speed, fmt="o" + ls, color=color, ms=3)
            helper.errorbars(axes[2, col], x, fluctuation, fmt="o" + ls, color=color, ms=3)
            diverged = np.array(diverged)
            for row in range(3 if diverged.any() else 0):  # mark diverged runs on the bottom axis line (y in axes coordinates)
                ax = axes[row, col]  # (an empty unclipped line would give constrained layout an unbounded box)
                ax.plot(x[diverged], np.zeros(diverged.sum()), "x", color=color, ms=5, clip_on=False, transform=ax.get_xaxis_transform())
            axes[2, col].set_xlabel(r"Recurrent / readout step size $\alpha_W / \alpha$")
            for row in range(3):
                axes[row, col].set_xscale("log")
                axes[row, col].axvline(DEFAULTS[credit], color="k", ls=ls, lw=0.6, alpha=0.5)
    axes[0, 0].set_ylabel(f"CR level, last {c.RNN_CONVERGENCE_TRIALS} trials")
    axes[1, 0].set_ylabel("Trials to 90% of final CR")
    axes[2, 0].set_ylabel(f"Relative SD of CR, last {c.RNN_CONVERGENCE_TRIALS} trials")
    axes[2, 0].set_yscale("log")
    note = f"\nmean and SD over {n_seeds} seeds" if n_seeds > 1 else ""
    axes[0, 0].legend(frameon=False, title=f"x: a seed diverged, vertical lines: defaults,\nleftmost point: frozen network{note}")
    helper.save_figure(fig, STUDY, "f4_rnn_step_size_summary")
