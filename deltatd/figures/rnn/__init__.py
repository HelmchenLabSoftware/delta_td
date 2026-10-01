"""Analyses of the learnable recurrent representation (RNN + TD(lambda) readout, plastic recurrent weights).

The plain "rnn" model (random initialization) is compared with the other models in the figures of the 2012 and 2008
studies. The modules here compare how the representation evolves from the four initializations (random, CSC,
microstimulus, presence) and scan the recurrent step size.
"""

from deltatd.figures.rnn import f1_initialization, f2_step_size  # noqa: F401

MODULES = (f1_initialization, f2_step_size)
