"""Response generation: the conditioned response is a thresholded leaky integration of the US prediction."""

from __future__ import annotations

from deltatd.utils import constants as c


class LeakyIntegratorResponse:
    """a_t = nu a_{t-1} + [V_t - theta]_+ (Eq. 2 of Ludvig et al. 2012), restarted from zero at every trial."""

    def __init__(self, threshold: float = c.RESPONSE_THRESHOLD, decay: float = c.RESPONSE_DECAY):
        self.threshold = threshold
        self.decay = decay
        self.a = 0.0

    def start_trial(self) -> None:
        self.a = 0.0

    def step(self, value: float) -> float:
        self.a = self.decay * self.a + max(value - self.threshold, 0.0)
        return self.a
