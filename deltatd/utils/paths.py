from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]  # Root directory of the project
RESULT_DIR = ROOT_DIR / "results"  # Directory for results
SIMULATION_DIR = RESULT_DIR / "simulation"  # Directory for simulation results
PLOT_DIR = ROOT_DIR / "figures"  # Directory for figures
DOC_DIR = ROOT_DIR / "docs"  # Directory for model notes


def result_path(study: str, experiment: str, name: str) -> Path:
    """Path of the simulation results of one run.

    :param study: study whose experiment is reproduced (e.g. ``"ludvig2012"``).
    :param experiment: experiment identifier (e.g. ``"acquisition"``).
    :param name: run identifier, typically ``"<representation>_<condition>"``.
    """
    path = SIMULATION_DIR / study / experiment / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def figure_path(study: str, name: str, extension: str = "pdf") -> Path:
    """Path of a figure file inside the figures folder of a study."""
    path = PLOT_DIR / study / f"{name}.{extension}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
