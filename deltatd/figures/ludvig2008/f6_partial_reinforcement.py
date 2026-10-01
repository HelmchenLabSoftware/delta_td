"""Partial reinforcement (Ludvig et al. 2008, Fig. 6): TD error on a rewarded and an omission trial after 500 trials
with reward probabilities 0, 0.25, 0.5, 0.75 and 1."""

from __future__ import annotations

from deltatd.figures import helper
from deltatd.figures.ludvig2008 import common
from deltatd.simulation import tasks
from deltatd.utils import constants as c
from deltatd.utils import ids

STUDY = common.STUDY
EXPERIMENT = ids.PARTIAL_REINFORCEMENT
CONDITIONS = {  # the reward schedule is random: the `seed` argument makes every model run once per seed
    f"p{p:.2f}": (lambda p=p, seed=c.SEED: tasks.partial_reinforcement_protocol(p, seed=seed)) for p in c.DA_REWARD_PROBABILITIES
}


def simulate(representations: list[str] = ids.ALL_REPRESENTATIONS) -> None:
    helper.simulate_conditions(
        STUDY, EXPERIMENT, CONDITIONS, common.representations_in(representations), model_factory=common.MODEL_FACTORY
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
            stats = helper.load_stats(STUDY, EXPERIMENT, representation, f"p{p:.2f}")
            trial = helper.probe_index(stats, label)
            time = common.seconds(stats, trial)
            mask = (time >= -0.5) & (time <= 2.5)
            mean, sd = stats.mean[ids.TD_ERROR][trial][mask], stats.sd[ids.TD_ERROR][trial][mask]
            axes[row, col].plot(time[mask], mean, color=helper.COLORS[representation])
            if stats.n > 1:
                axes[row, col].fill_between(time[mask], mean - sd, mean + sd, color=helper.COLORS[representation], alpha=helper.BAND_ALPHA, lw=0)
            if row == 0:
                helper.annotate_seeds(axes[row, col], stats)
            common.mark_events(axes[row, col], reward_s)
        axes[row, 0].set_ylabel(f"p = {p:.2f}\nTD error")
    for ax in axes[-1]:
        ax.set_xlabel("Time from cue onset (s)")
    helper.save_figure(fig, STUDY, "f6_partial_reinforcement")
