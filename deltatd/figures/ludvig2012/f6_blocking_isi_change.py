"""Blocking with a change of the ISI of the blocking stimulus between phases (Ludvig et al. 2012, Fig. 6)."""

from __future__ import annotations

from deltatd.figures import helper
from deltatd.figures.ludvig2012 import f5_blocking
from deltatd.simulation import tasks
from deltatd.utils import constants as c
from deltatd.utils import ids

STUDY = ids.LUDVIG2012
EXPERIMENT = ids.BLOCKING_ISI_CHANGE
CONDITION = "isi_change"
CONDITIONS = {CONDITION: lambda: tasks.blocking_protocol(*c.BLOCKING_CONDITIONS[CONDITION])}


def simulate(representations: list[str] = ids.ALL_REPRESENTATIONS) -> None:
    helper.simulate_conditions(STUDY, EXPERIMENT, CONDITIONS, representations)


def plot(representations: list[str] = ids.ALL_REPRESENTATIONS) -> None:
    """(a) CR levels on the probe trials; (b) CR time courses on the probe trials, one panel per representation."""
    helper.set_style()
    fig, axes = helper.representation_panels(representations, n_rows=2)
    ax_a = axes[0, 0]
    for ax in axes[0, 1:]:
        ax.remove()
    ax_a.set_title("Probe trials after ISI change")
    helper.grouped_bars(
        ax_a,
        f5_blocking.probe_levels(EXPERIMENT, CONDITION, representations),
        group_labels=ids.REPRESENTATION_LABELS,
        bar_labels=helper.PROBE_LABELS,
    )
    ax_a.legend(frameon=False)
    isi_phase1, isi_phase2, _ = c.BLOCKING_CONDITIONS[CONDITION]
    for ax, representation in zip(axes[1], representations):
        ax.set_title(ids.REPRESENTATION_LABELS[representation])
        results = helper.load(STUDY, EXPERIMENT, representation, CONDITION)
        for k, label in enumerate(helper.PROBE_LABELS):
            trial = helper.probe_index(results, label)
            ax.plot(helper.relative_time(results, trial), results[ids.RESPONSE][trial], color=f"C{k}", label=helper.PROBE_LABELS[label])
        ax.axvline(isi_phase2, color="k", ls=":", lw=0.8)
        ax.axvline(isi_phase1, color="k", ls="--", lw=0.8)
        ax.set_xlim(-10, isi_phase1 + 20)
        ax.set_xlabel("Time steps from CS onset")
    axes[1, 0].set_ylabel("CR level")
    axes[1, 0].legend(frameon=False)
    helper.save_figure(fig, STUDY, "f6_blocking_isi_change")
