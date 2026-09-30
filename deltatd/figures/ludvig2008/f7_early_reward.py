"""Early reward (Ludvig et al. 2008, Fig. 7): TD error and value on the first and last of 15 early-reward probes."""

from __future__ import annotations

import numpy as np

from deltatd.figures import helper
from deltatd.figures.ludvig2008 import common
from deltatd.simulation import simulate as sim
from deltatd.simulation import tasks
from deltatd.utils import constants as c
from deltatd.utils import ids

STUDY = common.STUDY
EXPERIMENT = ids.EARLY_REWARD
CONDITIONS = {"default": tasks.early_reward_protocol}


def simulate(representations: list[str] = ids.ALL_REPRESENTATIONS) -> None:
    helper.simulate_conditions(
        STUDY, EXPERIMENT, CONDITIONS, common.representations_in(representations), model_factory=sim.build_dopamine_model
    )


def plot(representations: list[str] = ids.ALL_REPRESENTATIONS) -> None:
    """First probe (solid) and last probe (dashed); TD error on top, value below."""
    helper.set_style()
    representations = common.representations_in(representations)
    fig, axes = helper.representation_panels(representations, n_rows=2)
    usual_s = c.DA_REWARD_DELAY / c.DA_STEPS_PER_SECOND
    early_s = c.DA_EARLY_REWARD_DELAY / c.DA_STEPS_PER_SECOND
    for col, representation in enumerate(representations):
        results = helper.load(STUDY, EXPERIMENT, representation, "default")
        probes = np.flatnonzero(results[ids.PROBE])
        for trial, style, label in ((probes[0], "-", "First early-reward probe"), (probes[-1], "--", "Last probe")):
            common.plot_error_and_value(axes[:, col], results, trial, color=helper.COLORS[representation], label=label, t_max=2.5)
            for ax in axes[:, col]:
                ax.lines[-1].set_linestyle(style)
        for ax in axes[:, col]:
            common.mark_events(ax, usual_s)
            ax.axvline(early_s, color="k", ls="--", lw=0.8)
        axes[1, col].set_xlabel("Time from cue onset (s)")
    axes[0, 0].set_ylabel("TD error")
    axes[1, 0].set_ylabel("Value")
    axes[0, 0].legend(frameon=False)
    helper.save_figure(fig, STUDY, "f7_early_reward")
