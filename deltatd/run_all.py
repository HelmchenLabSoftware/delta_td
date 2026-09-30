from deltatd.figures import ludvig2012
from deltatd.utils import ids


def run(representations: list[str] = ids.ALL_REPRESENTATIONS) -> None:
    """Run all simulations of all reproduced experiments."""
    print("Ludvig et al. (2012) experiments")
    for module in ludvig2012.MODULES:
        module.simulate(representations)
    print("All simulations complete!")
