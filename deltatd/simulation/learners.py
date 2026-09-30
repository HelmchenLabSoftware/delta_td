"""Learning rules that turn feature vectors into US predictions.

`TDLambda` is the linear TD(lambda) rule of Ludvig et al. (2012, Eqs. 1, 3-5). `DeltaTD` is the value-change
integration scheme of Schoenfeld et al. (2024, Supplementary Note 2, Eqs. 22-26): the network only learns the change
in value delta^C_t = w . x_t triggered by the current input, and the value estimate itself is carried forward in time
as V_t = V_{t-1} / gamma + delta^C_t until it is consumed by an expected reward (R_hat) or reset at trial end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deltatd.utils import constants as c
from deltatd.utils import ids


@dataclass
class StepOutput:
    value: float  # US prediction V_t used by the response model
    td_error: float  # Prediction error driving the weight update at this step


class Learner:
    """Base class. State (weights, traces) persists across trials unless `start_trial` says otherwise."""

    def __init__(self, n_features: int):
        self.n_features = n_features
        self.w = np.zeros(n_features)
        self.reset_traces()

    def reset_traces(self) -> None:
        raise NotImplementedError

    def start_trial(self) -> None:
        """Hook called at the first time step of every trial."""

    def step(
        self, x: np.ndarray, reward: float, us_present: bool, cs_offset: bool, trial_end: bool
    ) -> StepOutput:
        """Process the feature vector `x` and reward `reward` of the current time step and update the weights.

        :param us_present: whether a US is delivered at this time step.
        :param cs_offset: whether a conditioned stimulus turned off at this time step.
        :param trial_end: whether this is the last time step of the trial.
        """
        raise NotImplementedError


class TDLambda(Learner):
    """Linear TD(lambda) with accumulating eligibility traces.

    delta_t = r_t + gamma V(x_t) - V(x_{t-1});  e_t = gamma lambda e_{t-1} + x_{t-1};  w += alpha delta_t e_t.
    """

    def __init__(
        self,
        n_features: int,
        alpha: float = c.STEP_SIZE,
        gamma: float = c.DISCOUNT,
        lam: float = c.TRACE_DECAY,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        super().__init__(n_features)

    def reset_traces(self) -> None:
        self.e = np.zeros(self.n_features)
        self.x_prev = np.zeros(self.n_features)

    def step(
        self, x: np.ndarray, reward: float, us_present: bool, cs_offset: bool, trial_end: bool
    ) -> StepOutput:
        value = float(self.w @ x)
        td_error = reward + self.gamma * value - float(self.w @ self.x_prev)
        self.e = self.gamma * self.lam * self.e + self.x_prev
        self.w += self.alpha * td_error * self.e
        self.x_prev = x
        return StepOutput(value=value, td_error=td_error)


class DeltaTD(Learner):
    """Integrated value-change learner (Schoenfeld et al. 2024).

    At every step the input predicts a value change delta^C_t = w . x_t and the value is integrated as
    V_t = decay * V_{t-1} / gamma + delta^C_t. The weights are updated with the decomposed TD error (Eq. 8/24)
    w += eta (delta^U_t + gamma delta^C_t) z_{t-1} with delta^U_t = r_t - R_hat_t, where the imminent reward
    prediction R_hat_t equals V_t (the prediction is consumed and V restarts from zero) when the consuming event of
    the reset rule occurs and is zero otherwise. In the paper the consuming event was the lick action; here it is
    - `"offset"`: the termination of a conditioned stimulus (the event that also ends the prediction of the CSC and
      presence representations, and that coincides with the US time in delay conditioning),
    - `"us"`: the US delivery itself (hard-codes the reward timing),
    - `"trial_end"` / `"none"`: no consuming event.
    Except under `"none"`, the trial-end correction w += eta delta^reset z with delta^reset = -V (Eq. 26) is applied
    and V restarts from zero at every trial. The eligibility trace z_t = gamma lambda z_{t-1} + x_t is reset at trial
    start.

    The leaky-integration variant (decay < 1) mirrors the memory trace decay of the microstimulus representation:
    a persisting value estimate then produces a small negative error -(1 - decay) V_{t-1} at every step, so that
    persistence is no longer free and a learned negative weight on the US onset becomes the way to cancel it.

    :param reset_rule: how R_hat and the trial-end reset are determined (see ids.ResetRule).
    :param decay: leak of the integrated value estimate per time step (1 = perfect integration as in the paper).
    """

    def __init__(
        self,
        n_features: int,
        eta: float = c.DELTA_STEP_SIZE,
        gamma: float = c.DELTA_DISCOUNT,
        lam: float = c.DELTA_TRACE_DECAY,
        reset_rule: ids.ResetRule = c.DELTA_RESET_RULE,
        decay: float = c.DELTA_DECAY,
    ):
        self.eta = eta
        self.gamma = gamma
        self.lam = lam
        self.reset_rule = reset_rule
        self.decay = decay
        super().__init__(n_features)

    def reset_traces(self) -> None:
        self.z = np.zeros(self.n_features)
        self.v = 0.0

    def start_trial(self) -> None:
        self.z = np.zeros(self.n_features)
        if self.reset_rule != ids.RESET_NONE:
            self.v = 0.0

    def consumes(self, us_present: bool, cs_offset: bool) -> bool:
        """Whether the consuming event of the reset rule occurs at this time step."""
        if self.reset_rule == ids.RESET_ON_US:
            return us_present
        if self.reset_rule == ids.RESET_ON_OFFSET:
            return cs_offset
        return False

    def step(
        self, x: np.ndarray, reward: float, us_present: bool, cs_offset: bool, trial_end: bool
    ) -> StepOutput:
        delta_c = float(self.w @ x)
        value = self.decay * self.v / self.gamma + delta_c
        r_hat = value if self.consumes(us_present, cs_offset) else 0.0
        delta_u = reward - r_hat
        # With a leaky integrator the TD error r + gamma V_t - V_{t-1} picks up the decay term -(1 - decay) V_{t-1}
        td_error = delta_u + self.gamma * delta_c - (1.0 - self.decay) * self.v
        self.w += self.eta * td_error * self.z
        self.z = self.gamma * self.lam * self.z + x
        self.v = value - r_hat
        if trial_end and self.reset_rule != ids.RESET_NONE:
            delta_reset = -self.v
            self.w += self.eta * delta_reset * self.z
            td_error += delta_reset
            self.v = 0.0
        return StepOutput(value=value, td_error=td_error)
