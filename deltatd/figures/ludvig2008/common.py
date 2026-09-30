"""Shared pieces of the Ludvig et al. (2008) figure modules."""

from __future__ import annotations

import numpy as np

from deltatd.figures import helper
from deltatd.utils import constants as c
from deltatd.utils import ids

STUDY = ids.LUDVIG2008
REPRESENTATIONS: list[str] = list(c.DA_REPRESENTATIONS)


def representations_in(requested: list[str]) -> list[str]:
    """Restrict the study's models to those whose base representation was requested, keeping the study's order."""
    return [model for model in REPRESENTATIONS if ids.base_representation(model) in requested]


def seconds(results: dict[str, np.ndarray], trial: int) -> np.ndarray:
    """Time axis of a trial in seconds relative to the (first) cue onset."""
    return helper.relative_time(results, trial) / c.DA_STEPS_PER_SECOND


def plot_error_and_value(axes_row, results: dict[str, np.ndarray], trial: int, color: str, label: str, t_max: float = 3.0):
    """Plot the TD error (left axis) and value (right axis) of one trial against time in seconds."""
    time = seconds(results, trial)
    mask = (time >= -0.5) & (time <= t_max)
    axes_row[0].plot(time[mask], results[ids.TD_ERROR][trial][mask], color=color, label=label)
    axes_row[1].plot(time[mask], results[ids.VALUE][trial][mask], color=color, label=label)


def mark_events(ax, reward_time_s: float | None, cue_times_s: tuple[float, ...] = (0.0,)) -> None:
    for t in cue_times_s:
        ax.axvline(t, color="k", ls="-", lw=0.6, alpha=0.4)
    if reward_time_s is not None:
        ax.axvline(reward_time_s, color="k", ls=":", lw=0.8)
    ax.axhline(0, color="k", lw=0.4, alpha=0.4)
