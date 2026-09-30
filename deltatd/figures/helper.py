"""Shared plotting and bookkeeping helpers for the figure modules."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from deltatd.simulation import simulate, tasks  # noqa: E402
from deltatd.utils import ids, paths  # noqa: E402

COLORS: dict[str, str] = {
    ids.CSC: "#4C72B0",
    ids.MICROSTIMULUS: "#C44E52",
    ids.PRESENCE: "#55A868",
    ids.DELTA: "#8172B2",
}
PROBE_LABELS: dict[str, str] = {"compound": "CSA + CSB", "A_alone": "CSA alone", "B_alone": "CSB alone"}
PANEL_WIDTH = 3.2  # inches
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


def run_name(representation: str, condition: str) -> str:
    return f"{representation}_{condition}"


def simulate_conditions(
    study: str,
    experiment: str,
    conditions: dict[str, Callable[[], tasks.Protocol]],
    representations: Iterable[str] = ids.ALL_REPRESENTATIONS,
    verbose: bool = True,
) -> None:
    """Run every representation through every condition of an experiment and save the results.

    :param conditions: mapping of condition name to a factory returning the protocol of that condition.
    """
    for representation in representations:
        for condition, make_protocol in conditions.items():
            if verbose:
                print(f"  {experiment}: {representation} / {condition}", flush=True)
            model = simulate.build_model(representation)
            results = simulate.run_protocol(model, make_protocol())
            simulate.save_results(paths.result_path(study, experiment, run_name(representation, condition)), results)
    if verbose:
        print(f"  {experiment}: done")


def load(study: str, experiment: str, representation: str, condition: str) -> dict[str, np.ndarray]:
    return simulate.load_results(paths.result_path(study, experiment, run_name(representation, condition)))


def probe_index(results: dict[str, np.ndarray], label: str) -> int:
    """Index of the (last) probe trial carrying `label`."""
    matches = np.flatnonzero(results[ids.PROBE] & (results[ids.LABEL] == label))
    if matches.size == 0:
        raise KeyError(f"No probe trial labelled '{label}'")
    return int(matches[-1])


def relative_time(results: dict[str, np.ndarray], trial: int) -> np.ndarray:
    """Time axis of a trial relative to its earliest CS onset."""
    return np.arange(results[ids.VALUE].shape[1]) - results[ids.CS_ONSET][trial]


def representation_panels(
    representations: list[str], n_rows: int = 1, titles: list[str] | None = None
) -> tuple[plt.Figure, np.ndarray]:
    """Figure with one column per representation (and `n_rows` rows), titled by representation label by default."""
    fig, axes = plt.subplots(
        n_rows,
        len(representations),
        figsize=(PANEL_WIDTH * len(representations), PANEL_HEIGHT * n_rows),
        squeeze=False,
        sharey="row",
    )
    titles = titles or [ids.REPRESENTATION_LABELS[representation] for representation in representations]
    for ax, title in zip(axes[0], titles):
        ax.set_title(title)
    return fig, axes


def grouped_bars(
    ax: plt.Axes,
    data: dict[str, dict[str, float]],
    group_labels: dict[str, str] | None = None,
    bar_labels: dict[str, str] | None = None,
    ylabel: str = "CR level",
) -> None:
    """Grouped bar chart: one group per outer key (e.g. representation), one bar per inner key (e.g. probe type)."""
    groups = list(data)
    bars = list(next(iter(data.values())))
    width = 0.8 / len(bars)
    for b, bar in enumerate(bars):
        heights = [data[group][bar] for group in groups]
        offsets = np.arange(len(groups)) + (b - (len(bars) - 1) / 2) * width
        ax.bar(offsets, heights, width=width, label=(bar_labels or {}).get(bar, bar), color=f"C{b}")
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([(group_labels or {}).get(group, group) for group in groups])
    ax.set_ylabel(ylabel)
