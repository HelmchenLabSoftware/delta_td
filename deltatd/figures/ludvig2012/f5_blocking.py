"""Blocking with identical, later or earlier onset of the blocked stimulus (Ludvig et al. 2012, Fig. 5)."""

from __future__ import annotations

from deltatd.figures import helper
from deltatd.simulation import simulate as sim
from deltatd.simulation import tasks
from deltatd.utils import constants as c
from deltatd.utils import ids

STUDY = ids.LUDVIG2012
EXPERIMENT = ids.BLOCKING
CONDITION_NAMES = ("identical", "b_later", "b_earlier")
CONDITION_TITLES = {"identical": "Identical timing", "b_later": "Blocked CSB later", "b_earlier": "Blocked CSB earlier"}
CONDITIONS = {
    name: (lambda name=name: tasks.blocking_protocol(*c.BLOCKING_CONDITIONS[name])) for name in CONDITION_NAMES
}


def simulate(representations: list[str] = ids.ALL_REPRESENTATIONS) -> None:
    helper.simulate_conditions(STUDY, EXPERIMENT, CONDITIONS, representations)


def probe_level(label: str):
    """Metric: CR level on the (last) probe trial carrying `label`."""
    return lambda results: float(results[ids.CR_LEVEL][helper.probe_index(results, label)])


def probe_levels(experiment: str, condition: str, representations: list[str]) -> dict[str, dict[str, tuple[float, float]]]:
    """CR level (mean, SD over seeds) on the compound, CSA-alone and CSB-alone probe trials per representation."""
    levels = {}
    for representation in representations:
        stats = helper.load_stats(STUDY, experiment, representation, condition)
        levels[representation] = {label: stats.summary(probe_level(label)) for label in helper.PROBE_LABELS}
    return levels


def plot(representations: list[str] = ids.ALL_REPRESENTATIONS) -> None:
    helper.set_style()
    fig, axes = helper.representation_panels(
        list(CONDITION_NAMES), titles=[CONDITION_TITLES[condition] for condition in CONDITION_NAMES]
    )
    for ax, condition in zip(axes[0], CONDITION_NAMES):
        helper.grouped_bars(
            ax,
            probe_levels(EXPERIMENT, condition, representations),
            group_labels=ids.REPRESENTATION_LABELS,
            bar_labels=helper.PROBE_LABELS,
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside right upper", frameon=False, title=seed_note(representations))
    helper.save_figure(fig, STUDY, "f5_blocking")


def seed_note(representations: list[str]) -> str | None:
    """Legend note when some of the models shown are averaged over seeds."""
    stochastic = [r for r in representations if sim.is_stochastic(r)]
    return f"error bars: SD over {c.N_SEEDS} seeds\n({', '.join(ids.REPRESENTATION_LABELS[r] for r in stochastic)})" if stochastic else None
