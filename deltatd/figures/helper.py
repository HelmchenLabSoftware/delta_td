"""Shared plotting and bookkeeping helpers for the figure modules."""

from __future__ import annotations

import importlib
import inspect
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from deltatd.simulation import simulate, tasks  # noqa: E402
from deltatd.utils import constants as c  # noqa: E402
from deltatd.utils import ids, paths  # noqa: E402

COLORS: dict[str, str] = {
    ids.CSC: "#4C72B0",
    ids.MICROSTIMULUS: "#C44E52",
    ids.PRESENCE: "#55A868",
    ids.DELTA: "#8172B2",
    ids.DELTA_OFFSET: "#CC79A7",
    ids.RNN: "#937860",
    ids.RNN_CSC_INIT: "#4C72B0",
    ids.RNN_MICROSTIMULUS_INIT: "#C44E52",
    ids.RNN_PRESENCE_INIT: "#55A868",
    ids.RNN_EXACT: "#937860",
    ids.RNN_CSC_INIT_EXACT: "#4C72B0",
    ids.RNN_MICROSTIMULUS_INIT_EXACT: "#C44E52",
    ids.RNN_PRESENCE_INIT_EXACT: "#55A868",
}
LINESTYLES: dict[str, str] = {ids.RNN_LOCAL_CREDIT: "-", ids.RNN_EXACT_CREDIT: ":"}  # credit assignment of RNN models
PROBE_LABELS: dict[str, str] = {"compound": "CSA + CSB", "A_alone": "CSA alone", "B_alone": "CSB alone"}
PANEL_WIDTH = 3.2  # inches
BAND_ALPHA = 0.2  # Shading of the SD band around a mean over seeds
PANEL_HEIGHT = 2.4  # inches
DPI = 200


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.constrained_layout.use": True,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, study: str, name: str) -> None:
    """Save a figure as PDF (vector, for manuscripts) and PNG (for quick viewing)."""
    fig.savefig(paths.figure_path(study, name, "pdf"))
    fig.savefig(paths.figure_path(study, name, "png"), dpi=DPI)
    plt.close(fig)


def run_name(representation: str, condition: str, seed: int | None = None) -> str:
    """File stem of one run: `<representation>_<condition>` plus `_seed<k>` for the k-th repeat of a stochastic run."""
    return f"{representation}_{condition}" + ("" if seed is None else f"_seed{seed}")


def accepts_seed(make_protocol: Callable) -> bool:
    """Whether a protocol factory draws random numbers (it then takes the seed as keyword argument)."""
    try:
        return "seed" in inspect.signature(make_protocol).parameters
    except (TypeError, ValueError):
        return False


def seeds_of(model: str, make_protocol: Callable) -> tuple[int | None, ...]:
    """The seeds a run is repeated over: `constants.SEEDS` if the model or the protocol is stochastic, else (None,)."""
    return c.SEEDS if simulate.is_stochastic(model) or accepts_seed(make_protocol) else (None,)


@dataclass
class Job:
    """One simulation run: a model through the protocol of one condition of a figure module, saved under a name.

    The protocol factory is looked up in the figure module's `CONDITIONS` so that the job can be sent to another
    process. `name` is the first part of the file stem (usually the model id, a variant name for `fs1`).
    """

    study: str
    experiment: str
    module: str  # import path of the figure module whose CONDITIONS hold the protocol factory
    condition: str
    model: str
    name: str
    seed: int | None = None
    model_factory: str = "default"  # key of `simulate.MODEL_FACTORIES`
    representation_kwargs: dict = field(default_factory=dict)
    learner_kwargs: dict = field(default_factory=dict)
    run_kwargs: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.experiment}: {self.name} / {self.condition}" + ("" if self.seed is None else f" / seed {self.seed}")

    @property
    def cost(self) -> int:
        """Rough relative cost (exact-credit networks are ten times slower than the local rule, the rest is free)."""
        if not simulate.is_stochastic(self.model):
            return 1
        credit = ids.MODEL_VARIANTS.get(self.model, ids.ModelVariant(ids.RNN)).representation_kwargs.get("credit")
        return 100 if credit == ids.RNN_EXACT_CREDIT else 10

    def path(self):
        return paths.result_path(self.study, self.experiment, run_name(self.name, self.condition, self.seed))

    def run(self) -> None:
        make_protocol = importlib.import_module(self.module).CONDITIONS[self.condition]
        protocol = make_protocol(seed=self.seed) if accepts_seed(make_protocol) else make_protocol()
        factory = simulate.MODEL_FACTORIES[self.model_factory]
        model = factory(
            self.model, representation_kwargs=self.representation_kwargs, learner_kwargs=self.learner_kwargs, seed=self.seed
        )
        simulate.save_results(self.path(), simulate.run_protocol(model, protocol, **self.run_kwargs))


JOBS: list[Job] | None = None
"""When a list, `submit` collects the jobs instead of running them (used by `run_all` to run them in parallel)."""


def submit(job: Job) -> None:
    if JOBS is not None:
        JOBS.append(job)
    else:
        print(f"  {job.label}", flush=True)
        job.run()


def simulate_conditions(
    study: str,
    experiment: str,
    conditions: dict[str, Callable[..., tasks.Protocol]],
    representations: Iterable[str] = ids.ALL_REPRESENTATIONS,
    module: str | None = None,
    model_factory: str = "default",
    run_kwargs: dict | None = None,
) -> None:
    """Run (or queue) every representation through every condition of an experiment, repeated over the seeds.

    :param conditions: mapping of condition name to a factory returning the protocol of that condition; a factory
        with a `seed` keyword argument is a stochastic protocol and is run once per seed.
    :param module: import path of the figure module holding `conditions` as `CONDITIONS` (default: the caller).
    :param model_factory: key of `simulate.MODEL_FACTORIES` ("default": the 2012 parameters, "dopamine": 2008).
    :param run_kwargs: extra keyword arguments of `simulate.run_protocol` (e.g. which trials' features to record).
    """
    module = module or inspect.getmodule(inspect.stack()[1].frame).__name__
    for representation in representations:
        for condition, make_protocol in conditions.items():
            for seed in seeds_of(representation, make_protocol):
                submit(
                    Job(
                        study, experiment, module, condition, representation, representation, seed, model_factory,
                        run_kwargs=run_kwargs or {},
                    )
                )


def run_path(study: str, experiment: str, representation: str, condition: str, seed: int | None = None):
    return paths.result_path(study, experiment, run_name(representation, condition, seed))


FLOAT_KEYS: tuple[str, ...] = (
    ids.VALUE, ids.RESPONSE, ids.TD_ERROR, ids.CR_LEVEL, ids.PEAK_TIME, ids.DIMENSIONALITY, ids.WEIGHT_CHANGE
)
"""Recorded quantities that are averaged over seeds (the others describe the protocol and are identical)."""


def mask_diverged(results: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Copy of a run with NaN on and after the trial on which it diverged (that trial holds huge finite values)."""
    valid = valid_trials(results)
    if valid.all():
        return results
    masked = dict(results)
    for key in FLOAT_KEYS:
        if key not in results:  # older result files lack the later-added metrics
            continue
        values = results[key].astype(float, copy=True)
        values[~valid] = np.nan
        masked[key] = values
    return masked


def load_runs(study: str, experiment: str, representation: str, condition: str) -> list[dict[str, np.ndarray]]:
    """All saved runs of a model in a condition: one for a deterministic run, one per seed for a stochastic one."""
    single = run_path(study, experiment, representation, condition)
    if single.exists():
        return [mask_diverged(simulate.load_results(single))]
    seeded = [run_path(study, experiment, representation, condition, seed) for seed in c.SEEDS]
    found = [path for path in seeded if path.exists()]
    if not found:
        raise FileNotFoundError(f"No results for {run_name(representation, condition)} in {single.parent}")
    return [mask_diverged(simulate.load_results(path)) for path in found]


def load(study: str, experiment: str, representation: str, condition: str) -> dict[str, np.ndarray]:
    """The first run of a model in a condition (the deterministic run, or the first seed as an example)."""
    return load_runs(study, experiment, representation, condition)[0]


@dataclass
class Stats:
    """Mean and standard deviation over the seeds of the recorded quantities, plus the individual runs."""

    mean: dict[str, np.ndarray]
    sd: dict[str, np.ndarray]
    runs: list[dict[str, np.ndarray]]

    @property
    def n(self) -> int:
        return len(self.runs)

    @property
    def n_diverged(self) -> int:
        return sum(divergence_trial(run) is not None for run in self.runs)

    def __getitem__(self, key: str) -> np.ndarray:
        """Protocol descriptors (probe flags, labels, event times) and means are read like a single run."""
        return self.mean[key]

    def per_seed(self, metric: Callable[[dict[str, np.ndarray]], float]) -> np.ndarray:
        """A scalar metric evaluated on every run."""
        return np.array([metric(run) for run in self.runs], dtype=float)

    def summary(self, metric: Callable[[dict[str, np.ndarray]], float]) -> tuple[float, float]:
        """Mean and SD over the seeds of a scalar metric (NaN-aware; SD 0 for a single run)."""
        values = self.per_seed(metric)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return float(np.nanmean(values)), float(np.nanstd(values)) if self.n > 1 else 0.0

    @property
    def note(self) -> str:
        """Legend note on the statistics shown."""
        if self.n == 1:
            return ""
        note = f"mean $\\pm$ SD over {self.n} seeds"
        return note + (f", {self.n_diverged} diverged" if self.n_diverged else "")


def load_stats(study: str, experiment: str, representation: str, condition: str) -> Stats:
    runs = load_runs(study, experiment, representation, condition)
    mean, sd = dict(runs[0]), {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices of diverged runs
        for key in FLOAT_KEYS:
            if key not in runs[0]:
                continue
            stack = np.stack([run[key].astype(float) for run in runs])
            mean[key] = np.nanmean(stack, axis=0)
            sd[key] = np.nanstd(stack, axis=0) if len(runs) > 1 else np.zeros_like(mean[key])
    return Stats(mean, sd, runs)


def plot_band(ax: plt.Axes, x: np.ndarray, stats: Stats, key: str, trial: int | None = None, **kwargs):
    """Mean of a recorded quantity (a whole series or one trial of it) with a shaded band of one SD."""
    mean = stats.mean[key] if trial is None else stats.mean[key][trial]
    sd = stats.sd[key] if trial is None else stats.sd[key][trial]
    (line,) = ax.plot(x, mean, **kwargs)
    if stats.n > 1:
        ax.fill_between(x, mean - sd, mean + sd, color=line.get_color(), alpha=BAND_ALPHA, lw=0)
    return line


def annotate_seeds(ax: plt.Axes, stats: Stats) -> None:
    """Small note in the corner of a panel on the statistics shown (nothing for a deterministic run)."""
    if stats.note:
        ax.text(0.02, 0.98, stats.note, transform=ax.transAxes, va="top", ha="left", fontsize=6, color="0.4")


def errorbars(ax: plt.Axes, x, values: list[tuple[float, float]], fmt: str = "o-", **kwargs):
    """Points of (mean, SD) pairs with error bars (no bars for an SD of zero)."""
    means = np.array([m for m, _ in values])
    sds = np.array([s for _, s in values])
    return ax.errorbar(x, means, yerr=sds if np.any(sds > 0) else None, fmt=fmt, capsize=2, **kwargs)


def probe_index(results: dict[str, np.ndarray], label: str) -> int:
    """Index of the (last) probe trial carrying `label`."""
    matches = np.flatnonzero(results[ids.PROBE] & (results[ids.LABEL] == label))
    if matches.size == 0:
        raise KeyError(f"No probe trial labelled '{label}'")
    return int(matches[-1])


def relative_time(results: dict[str, np.ndarray], trial: int) -> np.ndarray:
    """Time axis of a trial relative to its earliest CS onset."""
    return np.arange(results[ids.VALUE].shape[1]) - results[ids.CS_ONSET][trial]


def trial_index(results: dict[str, np.ndarray], label: str) -> int:
    """Index of the (last) trial carrying `label`, probe or not."""
    matches = np.flatnonzero(results[ids.LABEL] == label)
    if matches.size == 0:
        raise KeyError(f"No trial labelled '{label}'")
    return int(matches[-1])


def representation_panels(
    representations: list[str], n_rows: int = 1, titles: list[str] | None = None, sharey: bool | str = "row"
) -> tuple[plt.Figure, np.ndarray]:
    """Figure with one column per representation (and `n_rows` rows), titled by representation label by default."""
    fig, axes = plt.subplots(
        n_rows,
        len(representations),
        figsize=(PANEL_WIDTH * len(representations), PANEL_HEIGHT * n_rows),
        squeeze=False,
        sharey=sharey,
    )
    titles = titles or [ids.REPRESENTATION_LABELS[representation] for representation in representations]
    for ax, title in zip(axes[0], titles):
        ax.set_title(title)
    return fig, axes


def grouped_bars(
    ax: plt.Axes,
    data: dict[str, dict[str, float | tuple[float, float]]],
    group_labels: dict[str, str] | None = None,
    bar_labels: dict[str, str] | None = None,
    ylabel: str = "CR level",
) -> None:
    """Grouped bar chart: one group per outer key (e.g. representation), one bar per inner key (e.g. probe type).

    A value may be a (mean, SD) pair, drawn as a bar with an error bar.
    """
    groups = list(data)
    bars = list(next(iter(data.values())))
    width = 0.8 / len(bars)
    for b, bar in enumerate(bars):
        values = [data[group][bar] for group in groups]
        heights = [v[0] if isinstance(v, tuple) else v for v in values]
        errors = [v[1] if isinstance(v, tuple) else 0.0 for v in values]
        offsets = np.arange(len(groups)) + (b - (len(bars) - 1) / 2) * width
        ax.bar(
            offsets, heights, width=width, label=(bar_labels or {}).get(bar, bar), color=f"C{b}",
            yerr=errors if any(e > 0 for e in errors) else None, capsize=2, error_kw={"lw": 0.8},
        )
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([(group_labels or {}).get(group, group) for group in groups])
    if len(groups) > 4:  # five model labels do not fit side by side in one panel
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor")
    ax.set_ylabel(ylabel)


def valid_trials(results: dict[str, np.ndarray]) -> np.ndarray:
    """Trials whose value time course is finite and bounded (False from the trial on which a run diverged)."""
    value = results[ids.VALUE]
    return np.all(np.isfinite(value) & (np.abs(value) <= simulate.DIVERGENCE_LIMIT), axis=1)


def per_trial(results: dict[str, np.ndarray], key: str) -> np.ndarray:
    """A per-trial series (CR level, dimensionality, ...) with NaN on and after the trial on which the run diverged."""
    return np.where(valid_trials(results), results[key], np.nan)


def divergence_trial(results: dict[str, np.ndarray]) -> int | None:
    """1-based trial on which the run diverged, or None."""
    invalid = np.flatnonzero(~valid_trials(results))
    return int(invalid[0]) + 1 if invalid.size else None


def response_width(results: dict[str, np.ndarray], trial: int) -> float:
    """Number of time steps during which the response of a trial exceeds half of its maximum (NaN without response)."""
    response = results[ids.RESPONSE][trial]
    peak = np.nanmax(response)
    return float(np.sum(response >= peak / 2)) if peak > 0 else np.nan


def trials_to_converge(cr_level: np.ndarray, fraction: float = 0.9, window: int = c.RNN_CONVERGENCE_TRIALS) -> float:
    """First trial (1-based) at which the CR level reaches `fraction` of its mean over the last `window` trials."""
    if not np.all(np.isfinite(cr_level)):
        return np.nan
    target = fraction * cr_level[-window:].mean()
    reached = np.flatnonzero(cr_level >= target)
    return float(reached[0] + 1) if reached.size else np.nan


def final_fluctuation(cr_level: np.ndarray, window: int = c.RNN_CONVERGENCE_TRIALS) -> float:
    """Relative standard deviation of the CR level over the last `window` trials (NaN for a diverged run)."""
    if not np.all(np.isfinite(cr_level)):
        return np.nan
    tail = cr_level[-window:]
    return float(tail.std() / tail.mean()) if tail.mean() > 0 else np.nan
