"""Evolution of the learned representation from each initialization and credit assignment rule.

Protocols: acquisition (ISI 25) and the timing set. Figures: the representation itself, one per credit rule
(US prediction time courses and the state of the network on the first and last trial of the first seed), and
summary metrics over trials (CR level, dimensionality of the state trajectory, weight change) plus the response
timing on probe trials (peak time and response width against the ISI), local credit solid, exact credit dotted. The
fixed representations that the initializations reproduce are simulated with the same protocols as references
(dashed). Every network is run once per seed (`constants.SEEDS`); lines are means, bands and error bars one SD.
"""

from __future__ import annotations

import numpy as np

from deltatd.figures import helper
from deltatd.simulation import tasks
from deltatd.utils import constants as c
from deltatd.utils import ids

STUDY = ids.RNN_STUDY
EXPERIMENT = ids.RNN_INITIALIZATION
MODELS = [model for models in ids.RNN_MODELS_BY_CREDIT.values() for model in models.values()]
REFERENCES = {init: init for init in ids.LUDVIG_REPRESENTATIONS}  # init -> fixed model reproducing it
ACQUISITION = "acquisition"
CONDITIONS = {
    ACQUISITION: lambda: tasks.acquisition_protocol(c.RNN_EXAMPLE_ISI),
    **{f"timing_isi{isi}": (lambda isi=isi: tasks.timing_protocol(isi)) for isi in c.TIMING_ISIS},
}
RECORDED_TRIALS = tuple(trial - 1 for trial in c.RNN_EXAMPLE_TRIALS)
RUN_KWARGS = {"record_features_at": RECORDED_TRIALS, "feature_metrics": True}
MIN_PROBE_RESPONSE = 0.1  # Mean probe responses below this are not normalized (they are noise)


def simulate(representations: list[str] = ids.ALL_REPRESENTATIONS) -> None:
    if ids.RNN not in representations:
        return
    helper.simulate_conditions(STUDY, EXPERIMENT, CONDITIONS, MODELS, run_kwargs=RUN_KWARGS)
    helper.simulate_conditions(STUDY, EXPERIMENT, CONDITIONS, ids.LUDVIG_REPRESENTATIONS, run_kwargs=RUN_KWARGS)


def plot(representations: list[str] = ids.ALL_REPRESENTATIONS) -> None:
    if ids.RNN not in representations:
        return
    helper.set_style()
    for credit, models in ids.RNN_MODELS_BY_CREDIT.items():
        plot_representation(credit, models)
    plot_metrics()


def sorted_activity(features: np.ndarray, window: slice, threshold: float = 0.01) -> np.ndarray:
    """Active units (rows) of a trial's features within a time window, normalized per unit and sorted by peak time."""
    activity = features[window].T  # (n_units, n_steps)
    peak = activity.max(axis=1)
    active = peak > threshold * peak.max()
    activity = activity[active] / peak[active, None]
    return activity[np.argsort(activity.argmax(axis=1), kind="stable")]


def plot_representation(credit: str, models: dict[str, str]) -> None:
    """Columns: initializations. Rows: US prediction over acquisition (mean over seeds), network state of the first
    seed on the first and last trial."""
    isi = c.RNN_EXAMPLE_ISI
    fig, axes = helper.representation_panels(
        list(models.values()), n_rows=3, sharey=False, titles=[f"{ids.RNN_INIT_LABELS[init]}, {ids.RNN_CREDIT_LABELS[credit]}" for init in models]
    )
    for ax in axes[0, 1:]:
        ax.sharey(axes[0, 0])
    for col, (init, model) in enumerate(models.items()):
        stats = helper.load_stats(STUDY, EXPERIMENT, model, ACQUISITION)
        ax = axes[0, col]
        if stats.n_diverged:
            ax.set_title(f"{ax.get_title()} ({stats.n_diverged}/{stats.n} seeds diverged)")
        for k, trial in enumerate(c.RNN_EXAMPLE_TRIALS):
            shade = 0.3 + 0.7 * k / max(len(c.RNN_EXAMPLE_TRIALS) - 1, 1)
            time = helper.relative_time(stats, trial - 1)
            helper.plot_band(ax, time, stats, ids.VALUE, trial - 1, color=helper.COLORS[model], alpha=shade, label=f"Trial {trial}")
        if init in REFERENCES:
            reference = helper.load(STUDY, EXPERIMENT, REFERENCES[init], ACQUISITION)
            ax.plot(time, reference[ids.VALUE][-1], "k--", lw=0.8, label=f"{ids.REPRESENTATION_LABELS[REFERENCES[init]]}, last trial")
        helper.annotate_seeds(ax, stats)
        ax.axvline(isi, color="k", ls=":", lw=0.8)
        ax.set_xlim(-10, 2 * isi + 10)
        ax.set_xlabel("Time steps from CS onset")
        ax.legend(frameon=False, loc="lower right")
        example = stats.runs[0]  # the network state is specific to a seed: show the first one
        onset = example[ids.CS_ONSET][0]
        window = slice(onset - 10, onset + 2 * isi + 10)
        recorded = list(example[ids.FEATURE_TRIALS])
        for row, trial in ((1, RECORDED_TRIALS[0]), (2, RECORDED_TRIALS[-1])):
            features = example[ids.FEATURES][recorded.index(trial)]
            if not np.all(np.isfinite(features)) or not np.any(features):  # diverged before or on this trial
                axes[row, col].set_title(f"State on trial {trial + 1}, seed {c.SEEDS[0]}: diverged", fontsize=8)
                axes[row, col].set_axis_off()
                continue
            activity = sorted_activity(features, window)
            axes[row, col].imshow(
                activity, aspect="auto", cmap="magma", extent=(-10, 2 * isi + 10, activity.shape[0], 0), interpolation="nearest"
            )
            axes[row, col].axvline(isi, color="w", ls=":", lw=0.8)
            axes[row, col].set_title(f"State on trial {trial + 1}, seed {c.SEEDS[0]} ({activity.shape[0]} active units)", fontsize=8)
            axes[row, col].set_xlabel("Time steps from CS onset")
    axes[0, 0].set_ylabel("US prediction")
    for row in (1, 2):
        axes[row, 0].set_ylabel("Units sorted by peak time")
    helper.save_figure(fig, STUDY, f"f1_rnn_initialization_representation_{credit}")


def plot_metrics() -> None:
    """Learning-curve metrics of the acquisition run and probe-trial timing of the timing set, per initialization."""
    fig, axes = helper.representation_panels(["learning", "peak", "probe"], n_rows=2, sharey=False, titles=["", "", ""])
    (ax_cr, ax_dim, ax_dw), (ax_peak, ax_width, ax_probe) = axes
    titles = {
        ax_cr: "CR level",
        ax_dim: "Dimensionality of the CS state",
        ax_dw: "Recurrent weight change",
        ax_peak: "Peak time, last probe trial",
        ax_width: "Response width, last probe trial",
        ax_probe: f"Last probe trial, ISI {c.TIMING_ISIS[-2]}",
    }
    for ax, title in titles.items():
        ax.set_title(title)
    n_seeds = 1
    for credit, models in ids.RNN_MODELS_BY_CREDIT.items():
        ls = helper.LINESTYLES[credit]
        for init, model in models.items():
            color = helper.COLORS[model]
            stats = helper.load_stats(STUDY, EXPERIMENT, model, ACQUISITION)
            n_seeds = max(n_seeds, stats.n)
            trials = np.arange(1, len(stats[ids.CR_LEVEL]) + 1)
            label = ids.REPRESENTATION_LABELS[model]
            if stats.n_diverged:
                label += f" ({stats.n_diverged}/{stats.n} seeds diverged)"
            helper.plot_band(ax_cr, trials, stats, ids.CR_LEVEL, color=color, ls=ls, label=label)
            helper.plot_band(ax_dim, trials, stats, ids.DIMENSIONALITY, color=color, ls=ls)
            helper.plot_band(ax_dw, trials, stats, ids.WEIGHT_CHANGE, color=color, ls=ls)
            plot_timing(ax_peak, ax_width, ax_probe, model, color, ls)
            if init in REFERENCES and credit == ids.RNN_LOCAL_CREDIT:
                reference = helper.load_stats(STUDY, EXPERIMENT, REFERENCES[init], ACQUISITION)
                ax_cr.plot(trials, reference[ids.CR_LEVEL], color=color, ls="--", lw=0.8)
                ax_dim.plot(trials, reference[ids.DIMENSIONALITY], color=color, ls="--", lw=0.8)
                plot_timing(ax_peak, ax_width, ax_probe, REFERENCES[init], color, "--")
    for ax in (ax_cr, ax_dim, ax_dw):
        ax.set_xlabel("Trials")
    ax_dim.set_yscale("log")
    ax_dim.set_ylabel("Participation ratio")
    ax_dw.set_ylabel(r"$\|W - W_0\|_F$")
    ax_dw.set_yscale("log")
    for ax in (ax_peak, ax_width):
        ax.set_xlabel("ISI")
        ax.set_xticks(c.TIMING_ISIS)
    ax_peak.plot(c.TIMING_ISIS, c.TIMING_ISIS, "k:", lw=0.8)
    ax_peak.set_ylabel("Peak time (steps from CS onset)")
    ax_width.set_ylabel("Steps above half-maximal response")
    ax_probe.set_xlabel("Time steps from CS onset")
    ax_probe.set_ylabel("Response (normalized)")
    ax_probe.set_xlim(-10, 2 * c.TIMING_ISIS[-2] + 10)
    ax_cr.set_ylabel("CR level")
    note = f"; bands and error bars: SD over {n_seeds} seeds" if n_seeds > 1 else ""
    fig.legend(loc="outside right upper", frameon=False, title=f"solid: local, dotted: exact,\ndashed: fixed reference{note}")
    helper.save_figure(fig, STUDY, "f2_rnn_initialization_metrics")


def last_probe(results: dict[str, np.ndarray]) -> int | None:
    """Index of the last probe trial of a run, None when the run diverged before the first probe."""
    probes = np.flatnonzero(results[ids.PROBE] & helper.valid_trials(results))
    return int(probes[-1]) if probes.size else None


def peak_on_last_probe(results: dict[str, np.ndarray]) -> float:
    last = last_probe(results)
    return float(results[ids.PEAK_TIME][last]) if last is not None else np.nan


def width_on_last_probe(results: dict[str, np.ndarray]) -> float:
    last = last_probe(results)
    return helper.response_width(results, last) if last is not None else np.nan


def plot_timing(ax_peak, ax_width, ax_probe, model: str, color: str, ls: str) -> None:
    """Peak time and width of the response on the last probe trial of every ISI (mean and SD over seeds)."""
    peaks, widths = [], []
    for isi in c.TIMING_ISIS:
        stats = helper.load_stats(STUDY, EXPERIMENT, model, f"timing_isi{isi}")
        peaks.append(stats.summary(peak_on_last_probe))
        widths.append(stats.summary(width_on_last_probe))
        if isi == c.TIMING_ISIS[-2] and (last := last_probe(stats.runs[0])) is not None:
            response = stats.mean[ids.RESPONSE][last]
            peak = np.nanmax(response)
            time = helper.relative_time(stats, last)
            if peak < MIN_PROBE_RESPONSE:  # no response to normalize (the random init at this ISI)
                ax_probe.plot(time, np.zeros_like(time), color=color, ls=ls)
            else:
                ax_probe.plot(time, response / peak, color=color, ls=ls)
                if stats.n > 1:
                    sd = stats.sd[ids.RESPONSE][last] / peak
                    ax_probe.fill_between(time, (response - sd) / peak, (response + sd) / peak, color=color, alpha=helper.BAND_ALPHA, lw=0)
            ax_probe.axvline(isi, color="k", ls=":", lw=0.8)
    helper.errorbars(ax_peak, c.TIMING_ISIS, peaks, fmt="o" + ls, color=color, ms=3)
    helper.errorbars(ax_width, c.TIMING_ISIS, widths, fmt="o" + ls, color=color, ms=3)
