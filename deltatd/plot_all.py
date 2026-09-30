from deltatd.figures import ludvig2012
from deltatd.utils import ids, paths


def plot(representations: list[str] = ids.ALL_REPRESENTATIONS) -> None:
    """Plot all figures of all reproduced experiments from the saved simulation results."""
    for module in ludvig2012.MODULES:
        print(f"  plotting {module.__name__.split('.')[-1]}", flush=True)
        module.plot(representations)
    print(f"All figures plotted to {paths.PLOT_DIR}")
