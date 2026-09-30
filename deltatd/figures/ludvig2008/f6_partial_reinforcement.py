"""Partial reinforcement (Ludvig et al. 2008, Fig. 6): TD error on a rewarded and an omission trial after 500 trials
with reward probabilities 0, 0.25, 0.5, 0.75 and 1."""

from __future__ import annotations

from deltatd.figures import helper
from deltatd.figures.ludvig2008 import common
from deltatd.simulation import simulate as sim
from deltatd.simulation import tasks
from deltatd.utils import constants as c
from deltatd.utils import ids

STUDY = common.STUDY
EXPERIMENT = ids.PARTIAL_REINFORCEMENT
CONDITIONS = {f"p{p:.2f}": (lambda p=p: tasks.partial_reinforcement_protocol(p)) for p in c.DA_REWARD_PROBABILITIES}


def simulate(representations: list[str] = ids.ALL_REPRESENTATIONS) -> None:
    helper.simulate_conditions(
        STUDY, EXPERIMENT, CONDITIONS, common.representations_in(representations), model_factory=sim.build_dopamine_model
    )


def plot(representations: list[str] = ids.ALL_REPRESENTATIONS) -> None:
    """Rows: reward probabilities. Columns: rewarded and omission test trial for each representation."""
    helper.set_style()
    representations = common.representations_in(representations)
    n_rows = len(c.DA_REWARD_PROBABILITIES)
    columns = [(representation, label) for representation in representations for label in ("rewarded", "omission")]
    titles = [f"{ids.REPRESENTATION_LABELS[r]}, {label}" for r, label in columns]
    fig, axes = helper.representation_panels([r for r, _ in columns], n_rows=n_rows, titles=titles, sharey=True)
    reward_s = c.DA_REWARD_DELAY / c.DA_STEPS_PER_SECOND
    for row, p in enumerate(c.DA_REWARD_PROBABILITIES):
        for col, (representation, label) in enumerate(columns):
            results = helper.load(STUDY, EXPERIMENT, representation, f"p{p:.2f}")
            trial = helper.probe_index(results, label)
            time = common.seconds(results, trial)
            mask = (time >= -0.5) & (time <= 2.5)
            axes[row, col].plot(time[mask], results[ids.TD_ERROR][trial][mask], color=helper.COLORS[representation])
            common.mark_events(axes[row, col], reward_s)
        axes[row, 0].set_ylabel(f"p = {p:.2f}\nTD error")
    for ax in axes[-1]:
        ax.set_xlabel("Time from cue onset (s)")
    helper.save_figure(fig, STUDY, "f6_partial_reinforcement")
