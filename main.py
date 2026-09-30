"""Command line entry point.

>> python main.py simulate   # run all simulations (results/simulation)
>> python main.py plot       # plot all figures from the saved results (figures/)
>> python main.py all        # both
Use --representations to restrict the models, e.g. `python main.py all --representations csc delta`.
"""

import argparse
import logging

import deltatd
from deltatd.utils import ids

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("simulate", "plot", "all"), nargs="?", default="plot")
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=ids.ALL_REPRESENTATIONS,
        default=ids.ALL_REPRESENTATIONS,
        help="models to simulate or plot (default: all)",
    )
    args = parser.parse_args()
    if args.command in ("simulate", "all"):
        deltatd.run_all.run(args.representations)
    if args.command in ("plot", "all"):
        deltatd.plot_all.plot(args.representations)
