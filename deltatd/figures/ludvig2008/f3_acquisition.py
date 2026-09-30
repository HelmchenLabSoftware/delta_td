"""Simple acquisition (Ludvig et al. 2008, Fig. 3): TD error and value on trials 1, 100 and 1000."""

from __future__ import annotations

from deltatd.figures import helper
from deltatd.figures.ludvig2008 import common
from deltatd.simulation import simulate as sim
from deltatd.simulation import tasks
from deltatd.utils import constants as c
from deltatd.utils import ids

STUDY = common.STUDY
EXPERIMENT = ids.DA_ACQUISITION
CONDITIONS = {"default": tasks.dopamine_acquisition_protocol}


def simulate(representations: list[str] = ids.ALL_REPRESENTATIONS) -> None:
    helper.simulate_conditions(
        STUDY, EXPERIMENT, CONDITIONS, common.representations_in(representations), model_factory=sim.build_dopamine_model
    )



def plot(representations: list[str] = ids.ALL_REPRESENTATIONS) -> None:
    """Rows: TD error and value. Columns: representations. Lines: trials 1, 100, 1000."""
    helper.set_style()
    representations = common.representations_in(representations)
    fig, axes = helper.representation_panels(representations, n_rows=2)
    reward_s = c.DA_REWARD_DELAY / c.DA_STEPS_PER_SECOND
    for col, representation in enumerate(representations):
        results = helper.load(STUDY, EXPERIMENT, representation, "default")
        for k, trial in enumerate(c.DA_EXAMPLE_TRIALS):
            shade = 0.35 + 0.65 * k / max(len(c.DA_EXAMPLE_TRIALS) - 1, 1)
            common.plot_error_and_value(
                axes[:, col], results, trial - 1, color=helper.COLORS[representation], label=f"Trial {trial}", t_max=2.5
            )
            for line in (axes[0, col].lines[-1], axes[1, col].lines[-1]):
                line.set_alpha(shade)
        for ax in axes[:, col]:
            common.mark_events(ax, reward_s)
        axes[1, col].set_xlabel("Time from cue onset (s)")
    axes[0, 0].set_ylabel("TD error")
    axes[1, 0].set_ylabel("Value")
    axes[0, 0].legend(frameon=False)
    helper.save_figure(fig, STUDY, "f3_acquisition")
