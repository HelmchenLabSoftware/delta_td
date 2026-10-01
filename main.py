"""Command line entry point.

>> python main.py simulate   # run all simulations (results/simulation)
>> python main.py plot       # plot all figures from the saved results (figures/)
>> python main.py all        # both
Use --representations to restrict the models and --studies to restrict the reproduced papers, e.g.
`python main.py all --representations csc delta --studies ludvig2008`. The study `rnn` holds the analyses of the
learnable recurrent representation (initializations and recurrent step size) and needs `rnn` among the models.
"""

import argparse
import logging
import os

# The models' matrices are small (300 x 300): one BLAS thread per process is fastest, and the parallel simulation
# workers would otherwise oversubscribe the cores with their thread pools. Must be set before numpy is imported.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import deltatd  # noqa: E402
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
    parser.add_argument(
        "--studies",
        nargs="+",
        choices=(ids.LUDVIG2012, ids.LUDVIG2008, ids.RNN_STUDY),
        default=(ids.LUDVIG2012, ids.LUDVIG2008, ids.RNN_STUDY),
        help="studies whose experiments to simulate or plot (default: all)",
    )
    parser.add_argument("--workers", type=int, default=None, help="parallel processes for the simulations (default: cores - 1)")
    args = parser.parse_args()
    if args.command in ("simulate", "all"):
        deltatd.run_all.run(args.representations, args.studies, workers=args.workers)
    if args.command in ("plot", "all"):
        deltatd.plot_all.plot(args.representations, args.studies)
