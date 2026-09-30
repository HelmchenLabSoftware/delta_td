from deltatd.figures import ludvig2008, ludvig2012
from deltatd.utils import ids

STUDIES = {ids.LUDVIG2012: ludvig2012, ids.LUDVIG2008: ludvig2008}


def run(representations: list[str] = ids.ALL_REPRESENTATIONS, studies: list[str] = tuple(STUDIES)) -> None:
    """Run all simulations of all reproduced experiments."""
    for study in studies:
        print(f"{study} experiments")
        for module in STUDIES[study].MODULES:
            module.simulate(representations)
    print("All simulations complete!")
