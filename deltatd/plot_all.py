from deltatd.figures import ludvig2008, ludvig2012
from deltatd.utils import ids, paths

STUDIES = {ids.LUDVIG2012: ludvig2012, ids.LUDVIG2008: ludvig2008}


def plot(representations: list[str] = ids.ALL_REPRESENTATIONS, studies: list[str] = tuple(STUDIES)) -> None:
    """Plot all figures of all reproduced experiments from the saved simulation results."""
    for study in studies:
        for module in STUDIES[study].MODULES:
            print(f"  plotting {study}/{module.__name__.split('.')[-1]}", flush=True)
            module.plot(representations)
    print(f"All figures plotted to {paths.PLOT_DIR}")
