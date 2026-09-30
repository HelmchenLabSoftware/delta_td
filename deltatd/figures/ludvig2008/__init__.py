"""Dopamine / TD-error experiments of Ludvig, Sutton & Kehoe (2008), Neural Computation 20:3034-3054.

The observable is the TD error (dopamine response) and the value, not a conditioned response. Parameters follow
Section 2 of the paper (`constants.DA_*`); the presence representation was not part of that study.
"""

from deltatd.figures.ludvig2008 import (  # noqa: F401
    f3_acquisition,
    f4_reward_omission,
    f6_partial_reinforcement,
    f7_early_reward,
    f8_multiple_cues,
)

MODULES = (f3_acquisition, f4_reward_omission, f6_partial_reinforcement, f7_early_reward, f8_multiple_cues)
