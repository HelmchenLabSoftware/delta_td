"""Acquisition set of Ludvig et al. (2012): Fig. 2 (US prediction time courses) and Fig. 3 (effect of the ISI)."""

from __future__ import annotations

import numpy as np

from deltatd.figures import helper
from deltatd.simulation import tasks
from deltatd.utils import constants as c
from deltatd.utils import ids

STUDY = ids.LUDVIG2012
EXPERIMENT = ids.ACQUISITION
CONDITIONS = {f"isi{isi}": (lambda isi=isi: tasks.acquisition_protocol(isi)) for isi in c.ACQUISITION_ISIS}


def simulate(representations: list[str] = ids.ALL_REPRESENTATIONS) -> None:
    helper.simulate_conditions(STUDY, EXPERIMENT, CONDITIONS, representations)


def plot(representations: list[str] = ids.ALL_REPRESENTATIONS) -> None:
    helper.set_style()
    plot_f2(representations)
    plot_f3(representations)


def plot_f2(representations: list[str]) -> None:
    """US prediction within a trial at several points of acquisition (ISI 25), one panel per representation."""
    isi = c.ACQUISITION_EXAMPLE_ISI
    fig, axes = helper.representation_panels(representations)
    for ax, representation in zip(axes[0], representations):
        results = helper.load(STUDY, EXPERIMENT, representation, f"isi{isi}")
        for k, trial in enumerate(c.ACQUISITION_EXAMPLE_TRIALS):
            time = helper.relative_time(results, trial - 1)
            shade = 0.3 + 0.7 * k / max(len(c.ACQUISITION_EXAMPLE_TRIALS) - 1, 1)
            ax.plot(time, results[ids.VALUE][trial - 1], color=helper.COLORS[representation], alpha=shade, label=f"Trial {trial}")
        ax.axvline(isi, color="k", ls=":", lw=0.8)
        ax.set_xlim(-10, 2 * isi + 10)
        ax.set_xlabel("Time steps from CS onset")
    axes[0, 0].set_ylabel("US prediction")
    axes[0, 0].legend(frameon=False)
    helper.save_figure(fig, STUDY, "f2_acquisition_value_timecourse")


def plot_f3(representations: list[str]) -> None:
    """(a) CR level on the last trial as a function of ISI; (b) learning curves for each ISI."""
    fig, axes = helper.representation_panels(representations, n_rows=2)
    ax_a = axes[0, 0]
    for ax in axes[0, 1:]:
        ax.remove()
    ax_a.set_title("CR level after %d trials" % c.N_TRIALS_ACQUISITION)
    for representation in representations:
        asymptotes = []
        for isi in c.ACQUISITION_ISIS:
            results = helper.load(STUDY, EXPERIMENT, representation, f"isi{isi}")
            asymptotes.append(results[ids.CR_LEVEL][-1])
        ax_a.plot(
            c.ACQUISITION_ISIS, asymptotes, "o-", color=helper.COLORS[representation], label=ids.REPRESENTATION_LABELS[representation]
        )
    ax_a.set_xlabel("Interstimulus interval (ISI)")
    ax_a.set_ylabel("CR level")
    ax_a.legend(frameon=False)

    for ax, representation in zip(axes[1], representations):
        ax.set_title(ids.REPRESENTATION_LABELS[representation])
        for k, isi in enumerate(c.TIMING_ISIS):
            results = helper.load(STUDY, EXPERIMENT, representation, f"isi{isi}")
            ax.plot(np.arange(1, len(results[ids.CR_LEVEL]) + 1), results[ids.CR_LEVEL], color=f"C{k}", label=f"ISI {isi}")
        ax.set_xlabel("Trials")
    axes[1, 0].set_ylabel("CR level")
    axes[1, 0].legend(frameon=False)
    helper.save_figure(fig, STUDY, "f3_acquisition_isi")
