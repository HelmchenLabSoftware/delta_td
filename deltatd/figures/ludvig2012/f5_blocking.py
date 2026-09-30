"""Blocking with identical, later or earlier onset of the blocked stimulus (Ludvig et al. 2012, Fig. 5)."""

from __future__ import annotations

from deltatd.figures import helper
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


def probe_levels(experiment: str, condition: str, representations: list[str]) -> dict[str, dict[str, float]]:
    """CR level on the compound, CSA-alone and CSB-alone probe trials for every representation."""
    levels = {}
    for representation in representations:
        results = helper.load(STUDY, experiment, representation, condition)
        levels[representation] = {
            label: float(results[ids.CR_LEVEL][helper.probe_index(results, label)]) for label in helper.PROBE_LABELS
        }
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
    axes[0, 0].legend(frameon=False)
    helper.save_figure(fig, STUDY, "f5_blocking")
