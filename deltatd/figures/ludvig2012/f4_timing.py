"""Timing set of Ludvig et al. (2012), Fig. 4: CR time course and peak time on unreinforced probe trials."""

from __future__ import annotations

import numpy as np

from deltatd.figures import helper
from deltatd.simulation import tasks
from deltatd.utils import constants as c
from deltatd.utils import ids

STUDY = ids.LUDVIG2012
EXPERIMENT = ids.TIMING
CONDITIONS = {f"isi{isi}": (lambda isi=isi: tasks.timing_protocol(isi)) for isi in c.TIMING_ISIS}


def simulate(representations: list[str] = ids.ALL_REPRESENTATIONS) -> None:
    helper.simulate_conditions(STUDY, EXPERIMENT, CONDITIONS, representations)


def plot(representations: list[str] = ids.ALL_REPRESENTATIONS) -> None:
    """Top row: response time course on the last probe trial. Bottom row: peak time across probe trials."""
    helper.set_style()
    fig, axes = helper.representation_panels(representations, n_rows=2, sharey=False)
    for ax in axes[1, 1:]:
        ax.sharey(axes[1, 0])
    for col, representation in enumerate(representations):
        for k, isi in enumerate(c.TIMING_ISIS):
            stats = helper.load_stats(STUDY, EXPERIMENT, representation, f"isi{isi}")
            probes = np.flatnonzero(stats[ids.PROBE])
            last = probes[-1]
            time = helper.relative_time(stats, last)
            helper.plot_band(axes[0, col], time, stats, ids.RESPONSE, last, color=f"C{k}", label=f"ISI {isi}")
            axes[0, col].axvline(isi, color=f"C{k}", ls=":", lw=0.8)
            peak_mean, peak_sd = stats.mean[ids.PEAK_TIME][probes], stats.sd[ids.PEAK_TIME][probes]
            axes[1, col].plot(probes + 1, peak_mean, ".", ms=3, color=f"C{k}")
            if stats.n > 1:
                axes[1, col].fill_between(probes + 1, peak_mean - peak_sd, peak_mean + peak_sd, color=f"C{k}", alpha=helper.BAND_ALPHA, lw=0)
            axes[1, col].axhline(isi, color=f"C{k}", ls=":", lw=0.8)
        helper.annotate_seeds(axes[0, col], stats)
        axes[0, col].set_xlim(-10, 2 * max(c.TIMING_ISIS) + 10)
        axes[0, col].set_xlabel("Time steps from CS onset")
        axes[1, col].set_xlabel("Trials")
        axes[1, col].set_ylim(0, c.TRIAL_DURATION - c.CS_ONSET_TIME)
    axes[0, 0].set_ylabel("CR level (probe trial, own scale)")
    axes[1, 0].set_ylabel("Peak time (steps from CS onset)")
    axes[0, 0].legend(frameon=False)
    helper.save_figure(fig, STUDY, "f4_timing_probe_trials")
