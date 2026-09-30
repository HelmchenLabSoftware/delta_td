"""Reward omission (Ludvig et al. 2008, Fig. 4): TD error and value on an omission trial after 1000 trials."""

from __future__ import annotations

from deltatd.figures import helper
from deltatd.figures.ludvig2008 import common
from deltatd.simulation import simulate as sim
from deltatd.simulation import tasks
from deltatd.utils import constants as c
from deltatd.utils import ids

STUDY = common.STUDY
EXPERIMENT = ids.REWARD_OMISSION
CONDITIONS = {"default": tasks.reward_omission_protocol}


def simulate(representations: list[str] = ids.ALL_REPRESENTATIONS) -> None:
    helper.simulate_conditions(
        STUDY, EXPERIMENT, CONDITIONS, common.representations_in(representations), model_factory=sim.build_dopamine_model
    )


def plot(representations: list[str] = ids.ALL_REPRESENTATIONS) -> None:
    """Omission trial (solid) against the last rewarded trial (faint), TD error on top and value below."""
    helper.set_style()
    representations = common.representations_in(representations)
    fig, axes = helper.representation_panels(representations, n_rows=2)
    reward_s = c.DA_REWARD_DELAY / c.DA_STEPS_PER_SECOND
    for col, representation in enumerate(representations):
        results = helper.load(STUDY, EXPERIMENT, representation, "default")
        omission = helper.probe_index(results, "omission")
        common.plot_error_and_value(axes[:, col], results, omission - 1, color="0.6", label="Last rewarded trial", t_max=3.0)
        common.plot_error_and_value(
            axes[:, col], results, omission, color=helper.COLORS[representation], label="Omission trial", t_max=3.0
        )
        for ax in axes[:, col]:
            common.mark_events(ax, reward_s)
        axes[1, col].set_xlabel("Time from cue onset (s)")
    axes[0, 0].set_ylabel("TD error")
    axes[1, 0].set_ylabel("Value")
    axes[0, 0].legend(frameon=False)
    helper.save_figure(fig, STUDY, "f4_reward_omission")
