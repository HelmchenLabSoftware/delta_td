# Model notes

Working notes on the four models compared in this repository, the mapping between the notation of
Ludvig, Sutton & Kehoe (2012) and Schoenfeld et al. (2024), and the modelling choices that are still open.

## Common frame: linear TD prediction of the US

All models produce a real-time US prediction $V_t$ and a conditioned response through the same thresholded
leaky integrator (Ludvig et al. 2012, Eq. 2):

$$a_t = \nu\, a_{t-1} + [V_t - \theta]_+ ,\qquad \theta = 0.25,\ \nu = 0.9 .$$

The *CR level* of a trial is $\max_t a_t$; the *peak time* is the time step of that maximum relative to CS onset.
Trials last 300 time steps, the US has magnitude 1 and lasts one step, a CS with interstimulus interval ISI turns on
ISI steps before the US and lasts ISI steps, i.e. it terminates when the US arrives (delay conditioning, CSs
coterminate; `CS_INCLUDES_US_STEP` keeps the CS on during the US step instead).

### Unreinforced trials in Ludvig et al. (2012)

The acquisition set (Fig. 2, 3) has none: every trial is reinforced and the CR is read on reinforced trials. All
other experiments contain *probe trials* without US, which is the only way to observe the response unconfounded by
the US and by CS termination at the US:

| experiment | unreinforced trials |
|---|---|
| timing set (Fig. 4) | every 5th of 500 trials, US omitted and CS extended to 2 x ISI (peak procedure) |
| blocking (Fig. 5, 6) | 3 probe trials after training: CSA alone, CSB alone, CSA + CSB |
| overshadowing (Fig. 7) | 3 probe trials after training: CSA alone, CSB alone, CSA + CSB |

Ludvig et al. (2008) additionally have explicit reward omission and partial reinforcement experiments.

## Ludvig et al. (2012): three temporal stimulus representations + TD($\lambda$)

$$V_t = \mathbf{w}_t^\top \mathbf{x}_t,\qquad
\delta_t = r_t + \gamma V_t(\mathbf{x}_t) - V_t(\mathbf{x}_{t-1}),\qquad
\mathbf{e}_{t} = \gamma\lambda\, \mathbf{e}_{t-1} + \mathbf{x}_{t-1},\qquad
\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha\, \delta_t\, \mathbf{e}_t$$

with $\gamma = 0.97$, $\lambda = 0.95$, $\alpha = 0.05$ (Appendix of the paper). The representations differ only in
$\mathbf{x}_t$:

| representation | elements per stimulus | $\mathbf{x}_t$ | temporal generalization |
|---|---|---|---|
| presence | 1 | $0.2$ while the stimulus is on | complete |
| complete serial compound (CSC) | one per time step (300) | unit element indexed by time since onset, while on | none |
| microstimulus (MS) | $m = 6$ | $x_t(i) = f(y_t, i/m, \sigma)\, y_t$ with trace $y_{t+1} = d\, y_t$, $y = 1$ at onset, $d = 0.985$, $\sigma = 0.08$, $f$ a Gaussian with prefactor $1/\sqrt{2\pi}$ | intermediate |

Every stimulus, including the US, spawns its own elements (needed for the MS representation to cancel the
post-US prediction; harmless for the other two).

## Schoenfeld et al. (2024), Supplementary Note 2: the $\Delta$-TD model

The value estimator does not read $V$ out of a temporal representation. It learns the *change* in value
caused by the current input and integrates it (Eq. 22):

$$\delta^C_t = \mathbf{w}^V_t \cdot \mathbf{x}_t,\qquad
\hat V_t = \gamma^{-1} \hat V_{t-1} + \delta^C_t - \gamma^{-1} \hat R_t .$$

$\hat R_t$ is the imminent reward prediction, obtained by *resetting* $\hat V$ when a reward is expected
(Eq. 25; in the paper: when the lick action is performed, $\hat R_{t+1} = \hat V_t$). The TD error decomposes
into an unconditioned and a conditioned part (Eq. 8),
$\delta^{TD}_t = \underbrace{(R_t - \hat R_t)}_{\delta^U_t} + \gamma\,\delta^C_t$, and the weights are
updated with an accumulating within-trial eligibility trace $\mathbf{z}_t = \mathbf{z}_{t-1} + \mathbf{x}_t$
(Eq. 24, written here without the dendritic gain factors) plus a trial-end correction (Eq. 26):

$$\Delta\mathbf{w}^V_t = \eta\,\big(\delta^U_t + \gamma\,\delta^C_t\big)\, \mathbf{z}_{t-1},\qquad
\Delta\mathbf{w}^V_{t_R} \mathrel{+}= \eta\, \delta^{reset}\, \mathbf{z},\quad \delta^{reset} = -\hat V_{t_R}.$$

Parameters in the paper: $\gamma = 1$, $\eta = 0.016$, 18 time steps per trial, inputs are one-step pulses of
the tone cue, the textures and the distractors.

### Where Ludvig's models get the reward timing from (checked numerically)

The sharp drop of the US prediction at the US is not produced by the learning rule but by two cues built into the
protocol and the representations:

1. **The CS coterminates with the US.** For the CSC and the presence representation the features simply vanish at
   the US step, so $V$ falls to zero for free.
2. **The US is itself a stimulus.** Its microstimuli learn strongly negative weights (after 200 acquisition trials
   with ISI 25: $w \approx (-1.9, -0.8)$ on the two fastest US microstimuli) and cancel the residual prediction of
   the CS microstimuli, which outlast the CS. Without US microstimuli the MS model has no drop at all
   ($V = 0.41$ at the US step, decaying slowly) and its response peak moves early (17 instead of 24).

Neither cue is a hard-coded reset; both are learnable inputs. For a fair comparison $\Delta$-TD receives the same
inputs.

### How it is implemented here (`learners.DeltaTD` on the `Onset` representation)

Defaults (`DELTA_*` in `constants.py`):

* input $\mathbf{x}_t$ = unit pulse at the onset of each stimulus **including the US** (`DELTA_US_AS_STIMULUS = True`);
  CS offsets can be added as learnable events too (`include_offsets`, off by default);
* $\gamma = 0.97$ (as the other models), $\lambda = 1$, $\eta = 0.05$ (Ludvig's step size);
* `reset_rule = "offset"`: CS termination consumes the prediction (see Decisions below). Alternatives: `"us"` (the
  paper's action-like reset moved to the US delivery), `"trial_end"` (only the trial-end correction of Eq. 26, which
  then teaches the US-onset weight $w_{US} \to -1$), `"none"` (no reset at all);
* optional leak `decay` of the integrator (mirrors the MS memory decay $d = 0.985$).

The variant table below was obtained with $\gamma = 1$ and `"trial_end"` unless stated.

Outcome of the variants on acquisition (ISI 25, 200 trials), the ISI-50 timing probe and blocking with earlier CSB:

| variant | $V$ at US $\to$ after US | probe trial (no US, CS extended) | comment |
|---|---|---|---|
| hard-coded US reset (paper) | $1 \to 0$ | $V$ stays at $w_A$ until trial end | reward timing given for free |
| US onset event + trial-end reset (**default**) | $w_{US} = -1$: $1 \to 0$ | $V$ stays at $w_A$ until trial end | same cue as Ludvig's US microstimuli |
| + CS offset events | $w_{off} \approx 0$ | $V$ drops when the CS turns off | offset never predicts anything on reinforced trials |
| US event, no reset, leak 0.985 | $w_{US} = -0.69$, $V(US^-) = 0.70$ | $V$ decays, response peaks early (17) | ISI dependence appears, resembles presence |
| US event, no reset, no leak | $w_A \to 10$, $V \to 10^3$ | diverges | persistence of $\hat V$ is free, nothing bounds it |

With $\gamma = 0.97$ and the US event the model reproduces the CSC's exponential ramp exactly on reinforced trials
($V(\text{onset}) = 0.48 \approx \gamma^{24}$, $V(US^-) = 1$, $w_{US} = -1.03$) but explodes on unreinforced probe
trials ($V \approx 900$ at trial end), which then wrecks the weights. The leak cannot fix this ($d/\gamma > 1$).

### The missing ingredient: action timing (Q-formulation)

In Schoenfeld et al. the reset is triggered by the lick, i.e. by the agent's own action, which is exactly what a
Pavlovian protocol lacks. The explosion with $\gamma < 1$ is the symptom: without an action there is no moment at
which the integrated value is "spent". The natural fix is to condition the estimate on the action, i.e. to learn a
Q-value: at every step the agent may respond ($a = \text{go}$, consume the prediction) or wait, with

$$\hat Q_t(\text{go}) = \hat V_t \ (\text{integrated as before}),\qquad
\hat R_t = \hat V_t\,[a_t = \text{go}],\qquad \delta^U_t = R_t - \hat R_t ,$$

so that responding too early yields a negative error (and a reset), responding at the US yields zero error, and not
responding lets the prediction ride (bounded by the US-onset event). Two ways to obtain the action:

* **Learned policy** (as in the paper, Eqs. 27-28): $\pi(\text{go} \mid \mathbf{x}_{1:t})$ trained by policy
  gradient on the reward; the response time of the model is then a learned quantity and the CR of the Ludvig response
  rule is replaced by the go probability. This is the go/no-go framing of the original task.
* **The CR as the action**: the thresholded leaky integrator already produces a response; its threshold crossing
  could be taken as the moment of action that consumes the prediction. This keeps the Ludvig response rule and closes
  the loop between prediction and response, but it is not what the paper did.

Either way the Ludvig models stay as they are: they never needed an action because their representation carries the
timing. Whether giving $\Delta$-TD an action counts as a fair condition is a judgement call: it does not receive the
US time as an input, it has to find it through its own behaviour.

### Decisions (2026-09-30)

* $\Delta$-TD must work for any $\gamma$ and without an action. The consuming event that replaces the lick is the
  **termination of the CS** (`reset_rule = "offset"`): $\hat R_t = \hat V_t$ when a CS turns off. This is the same
  information the CSC and presence representations use structurally (their features vanish at CS offset), and in
  delay conditioning it coincides with the US time. On reinforced trials $\delta^U = R - \hat R \to 0$ at
  convergence; on unreinforced trials $\delta^U = -\hat V$ (omission error, extinction) and the integrated value is
  consumed exactly, so nothing explodes. The US is still an input event (its weight stays near 0 under this rule
  because the offset already consumes the prediction).
* Why not the US onset weight alone: it cannot fire on omission trials, so with $\gamma < 1$ the value grows as
  $\gamma^{-t}$ until the trial ends ($\approx 900$). Partial cancellation by a learned offset weight does not help
  either: credit splits between the US onset and the CS offset (each $\approx -0.5$ when they coincide) and any
  residual regrows exponentially.
* $\gamma = 0.97$ for all four models. On reinforced trials $\Delta$-TD then reproduces the CSC's exponential ramp
  ($\hat V(\text{onset}) = \gamma^{ISI}$, $\hat V = 1$ at the US).
* The Q-formulation with a go/wait policy is **deferred**; it is the alternative consuming event if the CS-offset
  convention is judged unfair.
* Predictions that follow: $\Delta$-TD times its response off the CS offset, like the presence representation. On
  timing-set probes (CS extended to 2 x ISI) the value grows to $\gamma^{-ISI}$ before being consumed at 2 x ISI
  and the probes lower the asymptote (extinction on every 5th trial), as for presence. In asynchronous compounds the
  earlier CS takes the strength.

### Open questions

1. Whether to revisit the Q-formulation with an action model later, e.g. for the operant framing of the original
   task or if CS termination is considered to give $\Delta$-TD too much timing information.

## Reproduction status of Ludvig et al. (2012)

| figure | experiment | status |
|---|---|---|
| Fig. 2 | US prediction time course during acquisition (ISI 25) | reproduced qualitatively |
| Fig. 3 | CR level vs ISI, learning curves | reproduced qualitatively (CSC flat for long ISIs, inverted U for MS and presence) |
| Fig. 4 | CR timing on probe trials | reproduced qualitatively (presence peaks early at ISI 50, no CR at ISI 100) |
| Fig. 5 | blocking with identical / later / earlier CSB | reproduced qualitatively (full blocking for CSC and MS when CSB is later, CSB "steals" strength for presence, attenuated blocking for all when CSB is earlier) |
| Fig. 6 | blocking with ISI change | reproduced qualitatively; the secondary CSA peak of the CSC at the phase-1 US time (t = 100) only exists if the probe CSA stays on that long, so probe CSs are presented for the longest duration they were trained with (MS shows the secondary peak, at t ≈ 80, regardless because its trace outlasts the CS). Remaining discrepancy: the paper says the secondary peak is restricted to CSA-alone trials because "later temporal elements from CSB" acquire negative weights; here the compound probe shows it too, since CSB elements beyond its 25-step ISI are never active during training and cannot carry weights. This implies a protocol detail (CSB active after the US during training, or probe CSs extended) that the text does not specify. |
| Fig. 7 | overshadowing | reproduced qualitatively (CSC insensitive to CSA duration, MS and presence overshadow less with a longer CSA, two-peaked CSA response for CSC) |

Not reproduced so far: the faster learning of the presence representation with longer ISIs (Fig. 3b of the paper).

"Qualitatively" means the ordinal relationships described in the text hold; exact CR levels depend on the
undocumented details of the response integrator (reset per trial or not) and on the off-by-one conventions of the
TD update, which the paper does not fully specify.

## Ludvig et al. (2008): dopamine / TD-error experiments (not yet implemented)

Simple acquisition, reward omission, partial reinforcement, early reward and multiple cues, with the TD error
(rather than a CR) as the observable and parameters $\lambda = 0.95$, $\alpha = 0.01$, $\gamma = 0.98$, $n = 50$
microstimuli, $\sigma = 0.08$, 20 time steps per second, ITI 500 steps. The reward omission and early reward
experiments are the most diagnostic for $\Delta$-TD because its error at the usual US time depends entirely on the
reset rule.
