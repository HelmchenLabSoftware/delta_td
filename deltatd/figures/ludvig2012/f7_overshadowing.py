"""Overshadowing with synchronous and asynchronous compound stimuli (Ludvig et al. 2012, Fig. 7)."""

from __future__ import annotations

from deltatd.figures import helper
from deltatd.simulation import tasks
from deltatd.utils import constants as c
from deltatd.utils import ids

STUDY = ids.LUDVIG2012
EXPERIMENT = ids.OVERSHADOWING
CONDITIONS = {
    name: (lambda isi_a=isi_a: tasks.overshadowing_protocol(isi_a)) for name, isi_a in c.OVERSHADOWING_CONDITIONS.items()
}


def simulate(representations: list[str] = ids.ALL_REPRESENTATIONS) -> None:
    helper.simulate_conditions(STUDY, EXPERIMENT, CONDITIONS, representations)


def plot(representations: list[str] = ids.ALL_REPRESENTATIONS) -> None:
    """(a) synchronous overshadowing; (b) CR to CSB alone versus CSA duration; (c) time courses (asynchronous)."""
    helper.set_style()
    fig, axes = helper.representation_panels(representations, n_rows=2)
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    for ax in axes[0, 2:]:
        ax.remove()

    ax_a.set_title("Synchronous compound (same)")
    levels = {}
    for representation in representations:
        results = helper.load(STUDY, EXPERIMENT, representation, "same")
        levels[representation] = {
            label: float(results[ids.CR_LEVEL][helper.probe_index(results, label)]) for label in helper.PROBE_LABELS
        }
    helper.grouped_bars(ax_a, levels, group_labels=ids.REPRESENTATION_LABELS, bar_labels=helper.PROBE_LABELS)
    ax_a.legend(frameon=False)

    ax_b.set_title("Response to CSB alone")
    condition_names = list(c.OVERSHADOWING_CONDITIONS)
    for representation in representations:
        levels_b = []
        for condition in condition_names:
            results = helper.load(STUDY, EXPERIMENT, representation, condition)
            levels_b.append(results[ids.CR_LEVEL][helper.probe_index(results, "B_alone")])
        ax_b.plot(condition_names, levels_b, "o-", color=helper.COLORS[representation], label=ids.REPRESENTATION_LABELS[representation])
    ax_b.set_xlabel("Duration of overshadowing CSA")
    ax_b.set_ylabel("CR level")
    ax_b.legend(frameon=False)

    isi_a = c.OVERSHADOWING_CONDITIONS[c.OVERSHADOWING_EXAMPLE]
    for ax, representation in zip(axes[1], representations):
        ax.set_title(ids.REPRESENTATION_LABELS[representation])
        results = helper.load(STUDY, EXPERIMENT, representation, c.OVERSHADOWING_EXAMPLE)
        for k, label in enumerate(helper.PROBE_LABELS):
            trial = helper.probe_index(results, label)
            ax.plot(helper.relative_time(results, trial), results[ids.RESPONSE][trial], color=f"C{k}", label=helper.PROBE_LABELS[label])
        ax.axvline(isi_a, color="k", ls=":", lw=0.8)
        ax.set_xlim(-10, isi_a + 30)
        ax.set_xlabel("Time steps from CSA onset")
    axes[1, 0].set_ylabel("CR level")
    axes[1, 0].legend(frameon=False)
    helper.save_figure(fig, STUDY, "f7_overshadowing")
