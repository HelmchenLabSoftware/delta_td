"""Multiple cues (Ludvig et al. 2008, Fig. 8): two sequential cues before the reward, early and late in training,
with and without the second cue."""

from __future__ import annotations

from deltatd.figures import helper
from deltatd.figures.ludvig2008 import common
from deltatd.simulation import simulate as sim
from deltatd.simulation import tasks
from deltatd.utils import constants as c
from deltatd.utils import ids

STUDY = common.STUDY
EXPERIMENT = ids.MULTIPLE_CUES
CONDITIONS = {"default": tasks.multiple_cues_protocol}


def simulate(representations: list[str] = ids.ALL_REPRESENTATIONS) -> None:
    helper.simulate_conditions(
        STUDY, EXPERIMENT, CONDITIONS, common.representations_in(representations), model_factory=sim.build_dopamine_model
    )


def plot(representations: list[str] = ids.ALL_REPRESENTATIONS) -> None:
    """Rows: (both cues, second cue omitted) x (early, late). Columns: TD error and value per representation."""
    helper.set_style()
    representations = common.representations_in(representations)
    rows = [(k, kind) for k in c.DA_MULTI_EXAMPLE_TRIALS for kind in ("both", "omitted")]
    columns = [(representation, quantity) for representation in representations for quantity in ("TD error", "Value")]
    titles = [f"{ids.REPRESENTATION_LABELS[r]}: {q}" for r, q in columns]
    fig, axes = helper.representation_panels([r for r, _ in columns], n_rows=len(rows), titles=titles, sharey=False)
    second_s = c.DA_SECOND_CUE_DELAY / c.DA_STEPS_PER_SECOND
    reward_s = c.DA_MULTI_REWARD_DELAY / c.DA_STEPS_PER_SECOND
    for row, (k, kind) in enumerate(rows):
        for col, (representation, quantity) in enumerate(columns):
            results = helper.load(STUDY, EXPERIMENT, representation, "default")
            trial = helper.trial_index(results, f"{kind}_{k}")
            time = common.seconds(results, trial)
            mask = (time >= -0.5) & (time <= 4.0)
            key = ids.TD_ERROR if quantity == "TD error" else ids.VALUE
            axes[row, col].plot(time[mask], results[key][trial][mask], color=helper.COLORS[representation])
            common.mark_events(axes[row, col], reward_s, cue_times_s=(0.0, second_s))
            if kind == "omitted":
                axes[row, col].axvline(second_s, color="r", ls=":", lw=0.8)
        axes[row, 0].set_ylabel(f"{'both cues' if kind == 'both' else '2nd cue omitted'}\nafter {k} trials")
    for ax in axes[-1]:
        ax.set_xlabel("Time from first cue (s)")
    helper.save_figure(fig, STUDY, "f8_multiple_cues")
