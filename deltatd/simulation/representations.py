"""Temporal stimulus representations.

Each representation maps the stimuli that are present (and those that just turned on) at the current time step onto
a feature vector x_t that is fed to a learner. The three representations of Ludvig et al. (2012) differ in their
degree of temporal generalization: none for the complete serial compound, complete for the presence representation
and intermediate for the microstimuli. The onset representation only signals stimulus onsets and is the input to the
delta-TD learner, which integrates value changes instead of reading the value out of a temporal representation.
"""

from __future__ import annotations

import numpy as np

from deltatd.utils import constants as c
from deltatd.utils import ids


class Representation:
    """Base class of all stimulus representations.

    :param stimuli: identifiers of all stimuli that can occur, defining the stimulus indexing of `step`.
    """

    def __init__(self, stimuli: tuple[str, ...]):
        self.stimuli = tuple(stimuli)
        self.n_stimuli = len(self.stimuli)
        self.reset()

    @property
    def n_features(self) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        """Erase all memory of past stimuli (called once before a protocol, not between trials)."""

    def step(self, present: np.ndarray, onset: np.ndarray) -> np.ndarray:
        """Advance one time step and return the feature vector.

        :param present: boolean array (n_stimuli,) marking the stimuli that are on during this time step.
        :param onset: boolean array (n_stimuli,) marking the stimuli that turned on at this time step.
        :returns: feature vector of length `n_features`.
        """
        raise NotImplementedError

    def learn(self, td_error: float, readout: np.ndarray) -> None:
        """Adapt the representation to the TD error of the learner (no-op for the fixed representations).

        :param readout: current readout weights of the learner (the value is readout . x).
        """

    def weight_change(self) -> float:
        """Norm of the change of the representation's parameters since construction (0 for fixed representations)."""
        return 0.0

    def feature_labels(self) -> list[str]:
        raise NotImplementedError


class Presence(Representation):
    """One element per stimulus that is on (with a fixed salience) whenever the stimulus is present."""

    def __init__(self, stimuli: tuple[str, ...], salience: float = c.PRESENCE_SALIENCE):
        self.salience = salience
        super().__init__(stimuli)

    @property
    def n_features(self) -> int:
        return self.n_stimuli

    def step(self, present: np.ndarray, onset: np.ndarray) -> np.ndarray:
        return self.salience * present.astype(float)

    def feature_labels(self) -> list[str]:
        return list(self.stimuli)


class CompleteSerialCompound(Representation):
    """A distinct unit element for every time step since stimulus onset.

    :param gated_by_presence: elements are only active while the stimulus is on (Ludvig et al. 2012). With False the
        representation is a tapped delay line started by the onset that keeps ticking for `max_duration` steps
        regardless of the stimulus duration (the CSC of the dopamine models, Montague et al. 1996).
    """

    def __init__(
        self, stimuli: tuple[str, ...], max_duration: int = c.TRIAL_DURATION, gated_by_presence: bool = True
    ):
        self.max_duration = max_duration
        self.gated_by_presence = gated_by_presence
        super().__init__(stimuli)

    @property
    def n_features(self) -> int:
        return self.n_stimuli * self.max_duration

    def reset(self) -> None:
        self.age = np.full(self.n_stimuli, -1, dtype=int)  # Time steps since onset (-1 when not represented)

    def step(self, present: np.ndarray, onset: np.ndarray) -> np.ndarray:
        alive = present if self.gated_by_presence else np.ones_like(present)
        continuing = alive & ~onset & (self.age >= 0)
        self.age = np.where(onset, 0, np.where(continuing, self.age + 1, -1))
        x = np.zeros(self.n_features)
        active = (self.age >= 0) & (self.age < self.max_duration)
        x[np.flatnonzero(active) * self.max_duration + self.age[active]] = 1.0
        return x

    def feature_labels(self) -> list[str]:
        return [f"{s}_t{t}" for s in self.stimuli for t in range(self.max_duration)]


class Microstimulus(Representation):
    """Coarse coding of an exponentially decaying memory trace triggered by stimulus onset (Ludvig et al. 2008).

    Every stimulus (including the US) starts a trace y = 1 at onset that decays as y_{t+1} = d y_t. The trace height
    is encoded by m Gaussian basis functions centred at 1/m, 2/m, ..., 1, and each microstimulus level is the basis
    function value multiplied by the trace height (Eqs. 6-8 of Ludvig et al. 2012).
    """

    def __init__(
        self,
        stimuli: tuple[str, ...],
        n_microstimuli: int = c.N_MICROSTIMULI,
        sigma: float = c.MICROSTIMULUS_WIDTH,
        decay: float = c.MEMORY_DECAY,
    ):
        self.m = n_microstimuli
        self.sigma = sigma
        self.decay = decay
        self.centers = np.arange(1, self.m + 1) / self.m
        super().__init__(stimuli)

    @property
    def n_features(self) -> int:
        return self.n_stimuli * self.m

    def reset(self) -> None:
        self.trace = np.zeros(self.n_stimuli)

    def basis(self, y: np.ndarray) -> np.ndarray:
        """Gaussian basis functions f(y, mu, sigma) evaluated for every trace height y (n_stimuli, m)."""
        return np.exp(-((y[:, None] - self.centers[None, :]) ** 2) / (2 * self.sigma**2)) / np.sqrt(2 * np.pi)

    def step(self, present: np.ndarray, onset: np.ndarray) -> np.ndarray:
        self.trace = np.where(onset, 1.0, self.trace * self.decay)
        return (self.basis(self.trace) * self.trace[:, None]).ravel()

    def feature_labels(self) -> list[str]:
        return [f"{s}_ms{k + 1}" for s in self.stimuli for k in range(self.m)]


class Onset(Representation):
    """Unit pulse at the onset (and optionally the offset) of each stimulus (input of the delta-TD learner).

    Onsets and offsets are the sensory events that can trigger a learned change of the integrated value estimate.
    Whether the US counts as a stimulus is decided by the protocol runner (`run_protocol(us_as_stimulus=...)`).
    """

    def __init__(self, stimuli: tuple[str, ...], include_offsets: bool = c.DELTA_INCLUDE_OFFSETS):
        self.include_offsets = include_offsets
        super().__init__(stimuli)

    @property
    def n_features(self) -> int:
        return self.n_stimuli * (2 if self.include_offsets else 1)

    def reset(self) -> None:
        self.previous = np.zeros(self.n_stimuli, dtype=bool)

    def step(self, present: np.ndarray, onset: np.ndarray) -> np.ndarray:
        offset = self.previous & ~present
        self.previous = present.copy()
        if self.include_offsets:
            return np.concatenate([onset, offset]).astype(float)
        return onset.astype(float)

    def feature_labels(self) -> list[str]:
        labels = [f"{s}_onset" for s in self.stimuli]
        if self.include_offsets:
            labels += [f"{s}_offset" for s in self.stimuli]
        return labels


class RecurrentNetwork(Representation):
    """Learnable temporal representation: a rectified-linear recurrent network driven by stimulus onset pulses.

    h_t = [W h_{t-1} + U o_t]_+ where o_t are the onset pulses of all stimuli (including the US). Every stimulus is
    nominally assigned `n_units` units (the blocks of the structured initializations), but the recurrent weights W
    connect all units and are plastic. The feature vector seen by the TD learner is the whole state h_t.

    Plasticity of the recurrent weights (`learn`, called by the protocol runner after the learner's step) is a
    three-factor rule driven by the readout's TD error: E_t = gamma lambda E_{t-1} + dV_{t-1}/dW, W += alpha_W delta_t
    E_t, i.e. TD(lambda) applied to the recurrent synapses with the eligibility trace of the readout. The synaptic
    sensitivity dV_{t-1}/dW is either
    - `"local"`: w_i eps_ij with eps_ij,t = phi'_i,t h_j,t-1, the direct sensitivity of the postsynaptic unit to the
      synapse (presynaptic activity gated by the postsynaptic derivative) with the readout weight as the learning
      signal of the postsynaptic unit; credit only travels through time via the TD(lambda) trace E (cost n^2 per
      step). The e-prop self-connection term W_ii eps_ij,t-1 is deliberately left out: it explodes as W_ii^t for the
      autoregressive microstimulus block (diagonal entries up to 1.8) and inflates the eligibility of a
      self-sustaining presence unit linearly with time; or
    - `"exact"`: the semi-gradient propagated backwards through the network over the last `horizon` steps,
      sum_s c_s h_s^T with c_{t-2} = phi'_{t-1} w and c_{s-1} = phi'_s (W^T c_s) (cost 2 horizon n^2 per step).
    The onset weights U are fixed.

    :param init: "random" or one of the fixed representations that the initial dynamics reproduce (`initial_weights`).
    :param step_ratio: recurrent step size relative to the readout step size (`couple` sets the absolute value);
        defaults to `RNN_STEP_RATIO` or `RNN_EXACT_STEP_RATIO` depending on the credit assignment.
    :param noise: std of the Gaussian noise added to the structured initial weights.
    """

    def __init__(
        self,
        stimuli: tuple[str, ...],
        n_units: int = c.RNN_UNITS_PER_STIMULUS,
        init: str = c.RNN_INIT,
        step_ratio: float | None = None,
        noise: float = c.RNN_INIT_NOISE,
        credit: str = c.RNN_CREDIT,
        horizon: int = c.RNN_CREDIT_HORIZON,
        spectral_radius: float = c.RNN_RANDOM_SPECTRAL_RADIUS,
        seed: int = c.RNN_SEED,
    ):
        if step_ratio is None:
            step_ratio = c.RNN_STEP_RATIO if credit == ids.RNN_LOCAL_CREDIT else c.RNN_EXACT_STEP_RATIO
        self.n_units = n_units
        self.init = init
        self.step_ratio = step_ratio
        self.noise = noise
        self.credit = credit
        self.horizon = horizon
        self.spectral_radius = spectral_radius
        self.seed = seed
        self.n_total = n_units * len(stimuli)
        self.W, self.U = initial_weights(init, tuple(stimuli), n_units, noise, spectral_radius, seed)
        self.W0 = self.W.copy()
        self.couple(c.STEP_SIZE, c.DISCOUNT, c.TRACE_DECAY)
        super().__init__(stimuli)

    @property
    def n_features(self) -> int:
        return self.n_total

    @property
    def plastic(self) -> bool:
        return self.step_size > 0

    def couple(self, alpha: float, gamma: float, lam: float) -> None:
        """Tie the recurrent plasticity to the readout: step size ratio x alpha, eligibility decay gamma x lambda."""
        self.step_size = self.step_ratio * alpha
        self.trace_decay = gamma * lam

    def reset(self) -> None:
        n = self.n_total
        self.h = np.zeros(n)
        self.eps = np.zeros((n, n))  # Local eligibility eps_ij,t of the current step
        self.eps_prev = np.zeros((n, n))  # ... and of the previous step (sensitivity of V_{t-1})
        self.E = np.zeros((n, n))  # TD(lambda) eligibility trace of the recurrent synapses
        self.masks = np.zeros((self.horizon + 1, n))  # phi'(a_{t-k}), newest first (exact credit)
        self.pre = np.zeros((self.horizon + 1, n))  # h_{t-k-1}, the presynaptic activity of the same step

    def step(self, present: np.ndarray, onset: np.ndarray) -> np.ndarray:
        h_prev = self.h
        a = self.W @ h_prev + self.U @ onset.astype(float)
        mask = a > 0
        self.h = np.where(mask, a, 0.0)
        if self.plastic:
            if self.credit == ids.RNN_LOCAL_CREDIT:
                self.eps_prev = self.eps
                self.eps = mask[:, None] * h_prev[None, :]
            else:
                self.masks = np.roll(self.masks, 1, axis=0)
                self.pre = np.roll(self.pre, 1, axis=0)
                self.masks[0] = mask
                self.pre[0] = h_prev
        return self.h

    def sensitivity(self, readout: np.ndarray) -> np.ndarray:
        """dV_{t-1}/dW for the readout weights w (V = w . h), from the bookkeeping of the last `step`."""
        if self.credit == ids.RNN_LOCAL_CREDIT:
            return readout[:, None] * self.eps_prev
        # Exact chain: entry k of the buffers holds (phi'_{t-k}, h_{t-k-1}); V_{t-1} starts at k = 1
        credits = np.zeros((self.horizon, self.n_total))
        credit = self.masks[1] * readout
        for k in range(1, self.horizon + 1):
            credits[k - 1] = credit
            if k == self.horizon or not credit.any():
                break
            credit = self.masks[k + 1] * (self.W.T @ credit)
        return credits.T @ self.pre[1:]

    def learn(self, td_error: float, readout: np.ndarray) -> None:
        if not self.plastic:
            return
        self.E = self.trace_decay * self.E + self.sensitivity(readout)
        self.W += self.step_size * td_error * self.E

    def weight_change(self) -> float:
        return float(np.linalg.norm(self.W - self.W0))

    def feature_labels(self) -> list[str]:
        return [f"{s}_u{k}" for s in self.stimuli for k in range(self.n_units)]


def initial_weights(
    init: str, stimuli: tuple[str, ...], n_units: int, noise: float, spectral_radius: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Recurrent weights W (n_total, n_total) and onset weights U (n_total, n_stimuli) of an initialization.

    The structured initializations are block diagonal (one block per stimulus) and reproduce the corresponding fixed
    representation with the units of the block; the remaining units are silent apart from the initialization noise.
    """
    rng = np.random.default_rng(seed)
    n_total = n_units * len(stimuli)
    us = stimuli.index(ids.US)
    if init == ids.RNN_RANDOM_INIT:
        W = rng.standard_normal((n_total, n_total))
        W *= spectral_radius / np.abs(np.linalg.eigvals(W)).max()
        U = rng.standard_normal((n_total, len(stimuli)))
        U /= np.linalg.norm(U, axis=0, keepdims=True)
        return W, U
    W = noise * rng.standard_normal((n_total, n_total))
    U = np.zeros((n_total, len(stimuli)))
    if init == ids.CSC:
        block_w, block_u = delay_line_block(n_units)
    elif init == ids.MICROSTIMULUS:
        block_w, block_u = microstimulus_block(n_units)
    elif init == ids.PRESENCE:
        block_w, block_u = presence_block(n_units)
    else:
        raise ValueError(f"Unknown RNN initialization '{init}', expected one of {ids.RNN_INITS}")
    for k in range(len(stimuli)):
        sl = slice(k * n_units, (k + 1) * n_units)
        W[sl, sl] += block_w
        U[sl, k] = block_u
        if init == ids.PRESENCE and k != us:
            U[k * n_units, us] = -c.RNN_PRESENCE_INHIBITION  # The US switches the CS presence units off
        if init == ids.PRESENCE and k == us:
            W[k * n_units, k * n_units] = 0.0  # The US is on for a single step: no self-sustaining activity
    return W, U


def delay_line_block(n_units: int) -> tuple[np.ndarray, np.ndarray]:
    """Tapped delay line: the onset pulse enters unit 0 and is passed on to the next unit at every step (CSC)."""
    W = np.zeros((n_units, n_units))
    W[np.arange(1, n_units), np.arange(n_units - 1)] = 1.0
    U = np.zeros(n_units)
    U[0] = 1.0
    return W, U


def presence_block(n_units: int, salience: float = c.PRESENCE_SALIENCE) -> tuple[np.ndarray, np.ndarray]:
    """Self-sustaining unit 0 switched on by the onset pulse with the presence salience."""
    W = np.zeros((n_units, n_units))
    W[0, 0] = 1.0
    U = np.zeros(n_units)
    U[0] = salience
    return W, U


def microstimulus_block(
    n_units: int,
    order: int = c.RNN_MS_EMBEDDING_ORDER,
    hidden_scale: float = c.RNN_MS_HIDDEN_SCALE,
    ridge: float = c.RNN_MS_RIDGE,
    n_microstimuli: int = c.N_MICROSTIMULI,
    sigma: float = c.MICROSTIMULUS_WIDTH,
    decay: float = c.MEMORY_DECAY,
    n_steps: int = c.TRIAL_DURATION,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear system whose first `n_microstimuli` units follow the microstimulus levels after an onset.

    The microstimuli are not a linear dynamical system of their own dimension (Gaussian functions of a decaying
    trace), but they are well predicted by a linear function of their last `order` values (a vector autoregression
    fitted by ridge regression over a trial; the ridge keeps the weights of order one). The delayed copies occupy
    the units after the microstimuli, scaled by `hidden_scale` so that the readout mostly sees the microstimuli
    themselves. With a small ridge (1e-6) the reproduction is exact to about 1e-3 (feature maximum 0.4) but the
    fit is ill-conditioned (collinear lags, large cancelling coefficients) and the exact credit assignment diverges;
    the default ridge `RNN_MS_RIDGE` (1e-2) trades a reproduction error of about 0.06 for a stable block.
    """
    m = n_microstimuli
    if order * m > n_units:
        raise ValueError(f"The microstimulus init needs {order * m} units per stimulus, got {n_units}")
    ms = Microstimulus((ids.CS_A,), n_microstimuli=m, sigma=sigma, decay=decay)
    x = np.zeros((n_steps + order, m))  # Rows 0..order-1: before the onset (zeros)
    for t in range(n_steps):
        x[order + t] = ms.step(np.array([t == 0]), np.array([t == 0]))
    embedded = np.concatenate([x[order - k : n_steps + order - k] for k in range(order)], axis=1)  # (n_steps, order m)
    inputs, targets = embedded[:-1], embedded[1:, :m]
    coefficients = np.linalg.solve(inputs.T @ inputs + ridge * np.eye(order * m), inputs.T @ targets).T  # (m, order m)
    block = np.zeros((order * m, order * m))
    block[:m] = coefficients
    block[m:, :-m] = np.eye((order - 1) * m)  # The other units are delayed copies of the microstimuli
    scale = np.ones(order * m)
    scale[m:] = hidden_scale
    W = np.zeros((n_units, n_units))
    W[: order * m, : order * m] = (block * scale[:, None]) / scale[None, :]
    U = np.zeros(n_units)
    U[: order * m] = embedded[0] * scale
    return W, U
