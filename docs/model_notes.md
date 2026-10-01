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

## Reproducibility

All models are deterministic. The two random elements, the Bernoulli reward schedule of the partial reinforcement
experiment (`DA_PARTIAL_SEED`) and the random initial weights of the recurrent network plus the noise on its
structured initializations (`RNN_SEED`), each draw from a local `numpy.random.default_rng` seeded from
`constants.SEED` (0), so two simulations give bit-for-bit identical results (`tests/test_reproducibility.py`).
Every stochastic run (any recurrent network, including the structured initializations whose noise is random, and
every model in the partial reinforcement experiment whose schedule is random) is repeated over `constants.SEEDS`
(`N_SEEDS` = 5 seeds, 0 to 4). The figures show the mean over seeds with a band or error bar of one standard
deviation; a run that diverged contributes until the trial on which it diverged and the legend reports the number
of diverged seeds. Network states (unit activities) are shown for seed 0 only.

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
* `reset_rule = "event"`: the prediction is cashed in when the US arrives ($\delta^U = R - \hat V$, the ordinary TD
  comparison) or, if no US comes, when the CS terminates (bounds the prediction on unreinforced trials; the offset is
  not an input). Alternatives: `"offset"` (only the CS termination cashes in, the US is compared with nothing),
  `"us"` (only the US, unbounded on omission), `"trial_end"` (only the trial-end correction of Eq. 26, which then
  teaches the US-onset weight $w_{US} \to -1$), `"none"`;
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

* $\Delta$-TD must work for any $\gamma$ and without an action. The prediction is cashed in
  (`reset_rule = "event"`) **when the US arrives**, so that $\delta^U = R - \hat V$ is the ordinary TD comparison
  and an early reward yields the positive error left by the unfinished ramp, **or, when no US comes, when the CS
  terminates**, which bounds the integrated value on unreinforced trials ($\delta^U = -\hat V$, extinction). The CS
  offset is not an input of the model, the representation stays a brief onset pulse; it is the same information the
  CSC and presence representations use structurally (their features vanish at CS offset). In delay conditioning both
  events coincide. The US is still an input event (its weight stays near 0 because the cash-in already removes the
  prediction). Decided 2026-09-30 after comparing with the `"offset"`-only flavour on the 2008 early-reward task.
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

## Ludvig et al. (2008): dopamine / TD-error experiments

Same TD($\lambda$) machinery, the observable is the TD error $\delta_t$ (dopamine response) and the value, no
response rule. Parameters of Section 2 of the paper (`constants.DA_*`): $\lambda = 0.95$, $\alpha = 0.01$,
$\gamma = 0.98$, $m = 50$ microstimuli, $\sigma = 0.08$, $d = 0.985$, 20 time steps per second, trial onsets 500
steps apart (simulated as 500-step trials). The presence representation was not part of that study, so the figures
compare CSC, MS and $\Delta$-TD.

| figure | experiment | protocol |
|---|---|---|
| Fig. 3 | simple acquisition | cue at 0 s, reward at 1 s, 1000 trials; $\delta$ and $V$ on trials 1, 100, 1000 |
| Fig. 4 | reward omission | as above, reward omitted on trial 1000 |
| Fig. 6 | partial reinforcement | reward with p = 0, 0.25, 0.5, 0.75, 1 for 500 trials (seeded schedule), then one rewarded and one omission test trial |
| Fig. 7 | early reward | 1000 trials, then 15 probes with the reward at 0.5 s; first and last probe shown |
| Fig. 8 | multiple cues | cues at 0 s and 2 s, reward at 3 s; test trials with both cues and with the second cue omitted after 50 and 1000 trials |

Choices the paper leaves open (all in `constants.py`):

* **Cue duration.** The paper only gives cue onsets. Default `DA_CUE_LASTS_UNTIL_REWARD = True`: the cue stays on
  until the usual reward time (delay conditioning as in Fiorillo et al. 2003, and the same convention as the 2012
  tasks). With a one-step cue the MS model is unchanged (onset-triggered traces), the delay-line CSC is unchanged,
  but $\Delta$-TD with the CS-offset rule consumes its prediction one step after the cue and cannot predict the
  reward at all. **The 2008 tasks are therefore where the consumption rule of $\Delta$-TD is really tested**: on
  early-reward probes the reward arrives while the cue is still on, so under the `"offset"` rule the prediction is
  not consumed by the reward but by the cue offset at the usual time, producing a positive error at the early reward
  and a negative error at the usual time (like the CSC, unlike the data). The `"event"` rule (CS offset *or* US
  consumes) removes the negative error. This is the concrete question the variant figure is meant to inform.
* **CSC delay line.** The CSC of the dopamine models is a tapped delay line started by the cue onset
  (`gated_by_presence = False`), of length `DA_CSC_MAX_DURATION = 200` steps (10 s): long enough to cover every
  event, shorter than the ITI so that the reward's own delay line cannot predict the next trial's cue.
* **Partial reinforcement schedule.** Bernoulli per trial with a fixed seed; the two test trials are appended after
  training rather than picked from it.

### Findings (first pass, 2026-09-30)

The 2008 figures compare CSC, MS and $\Delta$-TD (default rule). The table also lists the earlier `"offset"`-only
flavour (available as the model variant `ids.DELTA_OFFSET`), which motivated the default.

| experiment | CSC | MS (paper's account) | $\Delta$-TD, only offset cashes in | $\Delta$-TD, US or offset (default) |
|---|---|---|---|---|
| acquisition | ramp, cue error 0.67, reward error vanishes | as paper (cue error, small extended errors, post-reward blip) | identical to CSC | identical to CSC |
| omission | sharp $-1$ at 1 s | shallow extended dip ($\approx 10\%$ of the cue error) | sharp $-1$ at CS offset (= 1 s) | same |
| partial reinforcement | cue error $\propto p$, reward error $\propto 1-p$ | same, extended omission dips | same as CSC, sharp omission dips | same |
| early reward | $+1$ at 0.5 s **and** $-1$ at 1 s on every probe | small dip at 1 s, gone by the last probe | $+1$ at 0.5 s and $-1$ at 1 s: the reward is not credited against the standing prediction because nothing consumes it at 0.5 s | $+0.2$ at 0.5 s, **nothing** at 1 s from the first probe on (Hollerman & Schultz) |
| multiple cues | late: error only at cue 1; cue 2 omitted: dip at 2 s, burst at 3 s | persistent error at cue 2, larger reward error when it is omitted | late: error only at cue 1; cue 2 omitted: **no error anywhere**, the integrator carries the prediction to the reward on its own (cue 2 never acquires weight) | same |

Two conclusions. First, cashing in at the US is what makes the reward comparison an ordinary TD error when the
reward and the CS termination decouple, and it changes nothing in the 2012 tasks (see the variant figure, first two
rows are identical); it is the default. Second, $\Delta$-TD's sharp, exactly timed omission and omission-like errors put it
in the CSC camp; the graded errors that made the MS model attractive for dopamine data are absent, and the multiple
cue result (no response to the reward after an omitted second cue) is a distinct, testable prediction.

## Supplementary figure: $\Delta$-TD consumption rules (`fs1_delta_variants`)

Rows are the rules `offset`, `event` (default), `us` and `trial_end` ($\gamma = 1$ for the last one), columns the US
prediction during acquisition, the response on a timing-set probe trial and the probe CR levels of blocking with an
earlier CSB.

## Learnable representation: recurrent network with a TD($\lambda$) readout (`rnn`)

The fourth kind of model does not fix the temporal representation but learns it. A rectified-linear recurrent
network of $N = 100$ units per stimulus (`RNN_UNITS_PER_STIMULUS`, the longest ISI of the experiments) is driven by
the onset pulses $\mathbf{o}_t$ of all stimuli (the US included, as for the Ludvig representations):

$$\mathbf{h}_t = \big[\mathbf{W}\,\mathbf{h}_{t-1} + \mathbf{U}\,\mathbf{o}_t\big]_+ ,\qquad V_t = \mathbf{w}^\top \mathbf{h}_t .$$

The readout $\mathbf{w}$ learns with the unchanged TD($\lambda$) rule of Ludvig et al., the onset weights
$\mathbf{U}$ are fixed, and the recurrent weights $\mathbf{W}$ are plastic. Nothing else changes: same response
rule, same protocols, same parameters. With a linear readout, every initialization spans the same function class
(any US prediction profile within the network's memory horizon), so the initializations are inductive biases for the
same TD learner, not different models. The question the `rnn` study asks is what TD learning makes of each of them.

### Plasticity of the recurrent weights

Three-factor rule driven by the TD error of the readout, in the form of TD($\lambda$) applied to the recurrent
synapses (`RecurrentNetwork.learn`):

$$\mathbf{E}_t = \gamma\lambda\,\mathbf{E}_{t-1} + \frac{\partial V_{t-1}}{\partial \mathbf{W}},\qquad
\mathbf{W} \mathrel{+}= \alpha_W\, \delta_t\, \mathbf{E}_t ,\qquad \alpha_W = \text{ratio} \times \alpha .$$

The synaptic sensitivity $\partial V_{t-1} / \partial W_{ij}$ is computed in one of two ways (`RNN_CREDIT`):

* `"local"` (default): $w_i\, \varepsilon_{ij,t-1}$ with $\varepsilon_{ij,t} = \phi'_{i,t}\, h_{j,t-1}$, the
  direct sensitivity of the postsynaptic unit to the synapse (presynaptic activity gated by the postsynaptic
  derivative); the learning signal of unit $i$ is its readout weight and credit only travels through time via the
  TD($\lambda$) trace $\mathbf{E}$. Cost $n^2$ per step (0.2 ms for $n = 300$). The e-prop self-connection term
  $W_{ii}\,\varepsilon_{ij,t-1}$ was tried first and dropped: it grows as $W_{ii}^t$, which explodes for the
  autoregressive microstimulus block (diagonal entries up to 1.8, divergence on the first trial at any step size)
  and inflates the eligibility of a self-sustaining presence unit linearly with time (divergence at ratios
  $\geq 10^{-2}$).
* `"exact"`: the semi-gradient propagated backwards through the network over the last `RNN_CREDIT_HORIZON` = 150
  steps, $\sum_s \mathbf{c}_s \mathbf{h}_s^\top$ with $\mathbf{c}_{t-2} = \phi'_{t-1} \odot \mathbf{w}$ and
  $\mathbf{c}_{s-1} = \phi'_s \odot (\mathbf{W}^\top \mathbf{c}_s)$. Verified against finite differences
  (`tests/test_rnn.py`); it coincides with the local rule when $\mathbf{W}$ is diagonal. Cost $2 K n^2$ per step
  (1.5 ms). It is the reference for what the local approximation misses; the `rnn` study runs every initialization
  with both rules (model variants `*_exact`), the 2012 and 2008 comparisons use the local rule only.

The ratio $\alpha_W / \alpha$ defaults to $2 \times 10^{-2}$ for the local rule (`RNN_STEP_RATIO`) and to
$5 \times 10^{-4}$ for the exact rule (`RNN_EXACT_STEP_RATIO`), both chosen from the scan of `RNN_STEP_RATIOS`
(0 and 12 values from $10^{-6}$ to $10^{-1}$) in the `rnn_step_size` figures, see the findings below. There is no
explicit stability control of $\mathbf{W}$; rectification bounds the activity from below only, and a run whose value
exceeds `DIVERGENCE_LIMIT` (or whose weights become NaN) is stopped and stored with NaN from that trial on.

### Initializations (`initial_weights`)

The structured initializations are block diagonal (one block per stimulus, remaining units silent) plus Gaussian
noise of std `RNN_INIT_NOISE` = $10^{-3}$ on every recurrent weight, without which the unused units and the
cross-stimulus weights would sit at a saddle point (no activity, no readout weight, hence no eligibility) forever.
They reproduce the fixed representations with the plasticity switched off (tested in `tests/test_rnn.py`):

| init | block | reproduction |
|---|---|---|
| `csc` | tapped delay line: the onset pulse enters unit 0 and is handed to the next unit every step | exact, equal to the ungated delay-line CSC of length 100 (the presence-gated CSC of 2012 differs only after the US, where the CS has ended anyway) |
| `presence` | unit 0 with a self-connection of 1, switched on by the onset with the presence salience 0.2 and switched off by a US onset weight of $-1$ (rectified to 0) | exact on reinforced trials; on unreinforced trials the unit stays on until the next US, because the CS offset is not an input of the network (a design decision, see below) |
| `microstimulus` | the 6 microstimuli are not a linear system of their own dimension (Gaussians of a decaying trace), but a 4th-order vector autoregression fitted by ridge regression over a trial reproduces them; the 18 delayed copies occupy further units at amplitude `RNN_MS_HIDDEN_SCALE` = 0.1 | with the ridge $10^{-2}$ adopted for stability under exact credit (see the diagnosis below): features within 0.06 (maximum 0.4) over the trial, frozen model value within 8% of the MS model; with ridge $10^{-6}$ the reproduction is exact to $1.4 \times 10^{-3}$ but the exact credit assignment diverges |
| `random` | Gaussian $\mathbf{W}$ over all 300 units scaled to spectral radius `RNN_RANDOM_SPECTRAL_RADIUS` = 1.4, unit-norm Gaussian onset vectors | rectification halves the effective gain (chaos threshold at $\sqrt 2$): at 1.2 the activity is gone after 20 steps, at 1.5 it explodes; at 1.4 the activity norm decays from 0.75 to 0.28 at 25 steps and 0.08 at 100 steps |

### Decisions (2026-09-30)

* Inputs are onset pulses only; the network has no presence input and no offset pulse. Presence-like features have
  to be produced by self-sustaining recurrence and switched off by the US. Consequence: on unreinforced trials the
  presence init stays on until the next US. Offset pulses could be added later if this is judged unfair.
* $N$ per stimulus is the longest ISI (100). The random init uses one pool of 300 units; the structured inits use
  one block per stimulus, but all recurrent weights are plastic, so cross-stimulus connections can be learned.
* Rectified linear units (the three structured inits are non-negative trajectories, so rectification leaves them
  unchanged and gives the presence init its switch-off).
* The plain `rnn` model (random init) is added to the 2012 and 2008 comparisons; the four initializations are
  compared in the `rnn` study only.

### Analyses (`figures/rnn`)

`f1_rnn_initialization_representation_{local,exact}`: US prediction over acquisition (ISI 25) per init, with the
fixed reference model's final prediction dashed, and the network state on the first and the last trial (active units
normalized and sorted by peak time), one figure per credit rule. `f2_rnn_initialization_metrics`: CR level,
dimensionality of the state trajectory during the CS (participation ratio of the Gram matrix; 25 for the CSC at
ISI 25, 1 for presence, in between for MS) and $\|\mathbf{W} - \mathbf{W}_0\|_F$ over trials, plus the timing set:
peak time and response width (steps above half the maximal response) on the last probe trial against the ISI, local
credit solid, exact credit dotted, fixed references dashed. The dimensionality and the response width against the
ISI are the two readouts of temporal generalization: a delay line keeps the dimensionality at the ISI and the width
constant, a scalar (Weber-law) representation widens the response with the ISI and keeps the dimensionality low.
`f3_rnn_step_size_{local,exact}`: the recurrent step-size scan per credit rule (CR level and dimensionality over
trials, last-trial prediction; diverged runs are cut at the divergence trial and marked). `f4_rnn_step_size_summary`:
final CR level, trials to 90% of it and its relative SD over the last 20 trials against the ratio, with the defaults.

### Findings (2026-09-30, revised 2026-10-01 with five seeds)

All numbers are means (± SD where it is not negligible) over the five seeds of `constants.SEEDS`, which change the
random weight matrix of the random init and the $10^{-4}$ noise of the structured inits. The single-seed numbers of
the first pass are superseded. The microstimulus init uses the ridge $10^{-2}$ decided in the diagnosis below.

Recurrent step-size scan (acquisition, ISI 25, 200 trials; entries: CR level over the last 20 trials, "div." = all
seeds diverged, "k/5 div." = k seeds diverged (the mean is over the others), "osc." = relative SD of the CR level
over the last 20 trials when above 1%):

| ratio $\alpha_W/\alpha$ | random, local | CSC, local | MS, local | presence, local | random, exact | CSC, exact | MS, exact | presence, exact |
|---|---|---|---|---|---|---|---|---|
| 0 (frozen) | 1.15 ± 0.56 | 5.29 | 4.54 | 3.96 | same | same | same | same |
| $10^{-6}$ | 1.15 ± 0.56 | 5.29 | 4.54 | 3.96 | 1.17 ± 0.56 | 5.29 | 4.56 | 3.98 |
| $10^{-5}$ | 1.15 ± 0.56 | 5.29 | 4.54 | 3.97 | 1.22 ± 0.54 | 5.29 | 4.71 | 4.09 |
| $10^{-4}$ | 1.15 ± 0.56 | 5.29 | 4.55 | 4.00 | 1.81 ± 0.64, osc. 2% | 5.29 | 4.79, osc. 15% | 4.88 ± 0.07 |
| $3 \times 10^{-4}$ | 1.14 ± 0.55 | 5.29 | 4.57 | 4.07 | 2.95 ± 1.10, osc. 2% | 5.29 | 5.01 ± 0.08 | 5.51 ± 0.42, 2/5 div. |
| $5 \times 10^{-4}$ (**exact default**) | 1.14 ± 0.54 | 5.29 | 4.58 | 4.14 | 3.76 ± 0.96, osc. 3% | 5.29 | 5.05, osc. 3% | 5.29 ± 0.18 |
| $10^{-3}$ | 1.13 ± 0.53 | 5.29 | 4.62 | 4.29 | 4.69 ± 0.41 | 5.29 | 5.05, 1/5 div. | 4.40 ± 1.93, osc. 97% |
| $3 \times 10^{-3}$ | 1.10 ± 0.50 | 5.29 | 4.73 | 4.71 | 5.01 ± 0.10, osc. 2% | 5.29 | 4.79, 3/5 div., osc. 18% | 5.31, 4/5 div. |
| $10^{-2}$ | 1.06 ± 0.41 | 5.29 | 4.92 | 5.33 ± 0.07 | 5.10 ± 0.09, osc. 9% | 5.29 | div. | div. |
| $2 \times 10^{-2}$ (**local default**) | 1.02 ± 0.30 | 5.29 | 4.99 | 5.58 ± 0.09 | 5.16, 1/5 div., osc. 10% | 5.29 | div. | div. |
| $3 \times 10^{-2}$ | 1.04 ± 0.23 | 5.29 | 5.01 | 5.65 ± 0.09 | 5.23, 1/5 div., osc. 10% | 5.29 | div. | div. |
| $5 \times 10^{-2}$ | 1.16 ± 0.27, osc. 3% | 5.29 | 5.02 | 5.57, 3/5 div. | 5.13, 3/5 div., osc. 18% | 5.29 | div. | div. |
| $10^{-1}$ | 1.33 ± 0.38, osc. 5% | 5.29 (dimensionality 12) | 5.04 | div. | div. | div. | div. | div. |

Criterion for the defaults, revised with seeds: the fastest ratio at which no seed of any initialization diverges.
Local: $2 \times 10^{-2}$ is confirmed (no divergence up to $3 \times 10^{-2}$; at $5 \times 10^{-2}$ three
presence seeds diverge, at $10^{-1}$ all). Exact: $5 \times 10^{-4}$ is the *only* ratio without a diverged seed
(two presence seeds already diverge at $3 \times 10^{-4}$, one MS seed at $10^{-3}$); the first-pass requirement of
a settled CR level (relative SD below 1%) cannot be met by all inits at any exact ratio (random 3%, MS 3%, presence
SD 0.18 across seeds at the default). The exact rule is therefore marginal on every init but the CSC, and the
stability boundary is not monotonic in the ratio (presence exact: 2/5 diverge at $3 \times 10^{-4}$, 0/5 at
$5 \times 10^{-4}$, oscillation at $10^{-3}$).

Initialization study at the default ratios (acquisition at ISI 25, 200 trials; timing set of 500 trials with
unreinforced probes every fifth trial; peak time / width of the response on the last probe trial, in steps from CS
onset; "--" = no response on the probe):

| model | CR (ISI 25) | ISI 10 | ISI 25 | ISI 50 | ISI 100 |
|---|---|---|---|---|---|
| CSC (fixed) | 5.29 | 9 / 12 | 24 / 18 | 49 / 18 | 99 / 18 |
| RNN, CSC init, local | 5.29 | 9 / 12 | 24 / 17 | 49 / 18 | 99 / 19 |
| RNN, CSC init, exact | 5.29 | 9 / 12 | 24 / 17 | 49 / 19 | 99 / 19 |
| microstimulus (fixed) | 4.67 | 11 / 18 | 26 / 28 | 48 / 25 | 101 / 55 |
| RNN, MS init, local | 5.02 | 10 / 16 | 24 / 25 | 49 / 32 | 98 / 47 |
| RNN, MS init, exact | 5.06 | 10 / 14 | 23 / 20 | 46 / 22 | -- (3/5 diverged) |
| presence (fixed) | 3.96 | 19 / 21 | 40 / 50 | 36 / 100 | -- |
| RNN, presence init, local | 5.58 ± 0.09 | 24 / 67 | 39 / 107 | 24 / 64 | -- |
| RNN, presence init, exact | 5.32 ± 0.21 | 9 / 20 | 15 / 36 | 20 / 52 | 120 ± 17 / 75 ± 7 |
| RNN, random init, local | 1.03 ± 0.29 | 9 / 12 | 17 ± 4 / 20 ± 2 | 44 / 15 (one seed) | -- |
| RNN, random init, exact | 3.90 ± 0.92 | 9 / 12 | 23 ± 1 / 20 ± 1 | 43 ± 5 / 26 ± 5 | -- |

* **Exact credit turns the random init into a competent but seed-dependent learner, the local rule does not.**
  With the local rule the random pool stays a slow learner at every ratio (CR 1.0 to 1.3, the spread of ± 0.5
  between seeds being larger than any effect of the ratio; the value outlasts the US because the US-driven activity
  of the pool is not cancelled). With exact credit at the default it reaches CR 3.9 ± 0.9 after 200 trials (4.7 at
  ratio $10^{-3}$, 5.1 at $3 \times 10^{-3}$), the value peaks before the US and is cancelled after it, and the
  response peaks at the US; how fast depends on the seed (trials to 90% of the final CR: 163 ± 17 at the default).
  The difference is credit assignment through the recurrent dynamics: the local rule only strengthens synapses whose
  postsynaptic unit already has a readout weight, so it cannot build the multi-step memory a random reservoir lacks.
* **The structured inits keep their temporal signature, independently of the seed.** The CSC init stays a delay
  line under both rules (peak at the US, width 18 steps at every ISI, SD over seeds zero to two decimals), although
  its dimensionality drops from 25 to 22 (local) as plasticity smears the line. The MS init keeps the Weber-like
  widening of the microstimuli (width 16 to 47 steps from ISI 10 to 100 with the local rule) while learning faster
  than the fixed microstimuli (5.02 against 4.67) and losing dimensionality (2.0 to 1.8). Plasticity sharpens the
  prediction on the trained ISI without changing the family of the representation.
* **The presence init is the one that plasticity changes qualitatively, and the two rules change it differently.**
  In plain acquisition both rules grow the self-connection past 1 and turn the plateau into the exponential ramp of
  TD (value at the US after 200 trials 1.06 ± 0.02 with the local rule, 0.98 ± 0.06 with exact credit; the local
  rule overshoots the CSC's 1.0 because a single exponential cannot also fit the onset step); the block keeps a
  single active unit under both rules. In the timing set, where every fifth trial is
  an unreinforced probe on which the unit is never switched off (no US, and the CS offset is not an input), the two
  rules end up differently. With the local rule the unit stays on and decays slowly (probe value at ISI 25: 0.35 at
  onset, 0.11 at the end of the trial), so the response is a plateau of 64 to 107 steps peaking between 24 and 39
  steps whatever the ISI, and at ISI 100 the plateau (0.21) never reaches the response threshold; the seeds agree
  to the step. With exact credit the activity on probes decays within the ISI at ISI 10 to 50 (0.2 to 0.03 within
  50 steps at ISI 50), so the response is early (peak 9, 15, 20 steps) and narrower (20 to 52 steps), whereas at
  ISI 100 the unit ramps up and the response peaks late (120 ± 17 steps, width 75 ± 7). Under exact credit the
  presence init is also the least stable: in acquisition three of the five seeds collapse from CR 5.8 to zero
  between trials 95 and 107 and recover to 5.2 to 5.3 by trial 150 (the SD band in the metrics figure; the two
  other seeds are smooth), and the scan shows divergence on both sides of the default ratio. Neither rule yields the fixed presence model's behaviour (peak at the CS offset, then decay).
* **The random init generalizes only within the memory horizon of the reservoir.** With the local rule it responds
  at ISI 10 (peak 9, width 12, CR 3.0 ± 0.1 on the last probe) and weakly at ISI 25 (CR 1.2 ± 0.5), and at ISI 50
  only one of five seeds responds at all (the pool's activity has decayed to 0.28 at 25 steps, 0.08 at 100). Exact
  credit extends the horizon to ISI 50 (peak 43 ± 5, width 26 ± 5, CR 4.2) but not to 100. In the 2012 and 2008
  figures (local rule) the random init therefore learns only the short ISIs (CR after 200 trials: 2.6 ± 0.1 at
  ISI 5, 3.1 ± 0.4 at ISI 10, 1.0 ± 0.3 at ISI 25, 0.1 ± 0.2 at ISI 50, 0 at ISI 100), barely responds in the
  blocking (blocking CS at ISI 50) and overshadowing (ISI 25) protocols (CR at most 1.3 ± 0.6, so these experiments
  are uninformative for it), and shows the TD-error features of Ludvig 2008 (reward omission dip, partial
  reinforcement, early reward) only qualitatively, with a noisy value that outlasts the US because the US-driven
  activity of the pool is not cancelled. One of the five seeds diverges on the omission trial of the 2008
  reward-omission experiment (see "Notes for later work").
* **Speed of the CR against the fixed references** at the defaults: every structured init learns at least as fast as
  the representation it reproduces (CR after 200 trials: CSC 5.29 = 5.29, MS 5.02 > 4.67, presence 5.58 > 3.96),
  because plasticity adds the exponential ramp the readout alone cannot express on presence-like features.
* **Variance across seeds** is negligible for the structured inits with the local rule (SD at most 0.09 in the CR
  level) and large for the random init (SD 0.3 to 0.9) and for every init at the stability boundary of the exact
  rule; conclusions about the random init rest on the means over five weight matrices, not on one.

## Diagnosis: the microstimulus init diverges under exact credit (2026-10-01)

**Symptom.** With the exact (backward-propagated) credit assignment the microstimulus init diverges on trial 2 at the
default ratio $5 \times 10^{-4}$ and at every ratio above $10^{-6}$; the other three inits are stable there, and the
same init is stable under the local rule at ratios up to $10^{-1}$.

**Cause: the block is a nearly singular autoregression, and the exact gradient travels through its raw
coefficients.** The microstimulus block is a companion-form realisation of a 4th-order vector autoregression fitted
to the six microstimuli (24 units: the microstimuli and three delayed copies). The microstimuli are smooth, so their
lagged values are almost collinear and the least-squares fit (ridge $10^{-6}$) picks large coefficients of
alternating sign that cancel in the forward pass. The forward dynamics are fine (spectral radius 0.97, exact
reproduction to $1.4 \times 10^{-3}$), but the matrix is strongly non-normal: $\|\mathbf{W}\|_2 = 18.5$ with
entries up to 9.4 (the hidden-copy scaling of 0.1 multiplies the copy coefficients by ten), against a spectral
radius below 1. Backward propagation uses $\mathbf{W}^\top$ step by step and sees the raw coefficients, not their
cancelling sum: $\|(\mathbf{W}^\top)^k\|_2$ reaches 71 at $k = 5$ and is still 10 at $k = 100$. Measured on trial 1,
20 steps after CS onset, the exact sensitivity $\|\partial V / \partial \mathbf{W}\|_F$ is 26 against 0.06 for the
local rule (a factor 400), concentrated in the rows of the hidden-copy units. The weight update of the first trial
is therefore already large enough to destabilise the block. The local rule never sees $\mathbf{W}^\top$; its
eligibility is bounded by the activities, which are smooth and small.

What does *not* help: the credit horizon (10 or 30 instead of 150: the amplification happens within the first
five backward steps), the hidden-unit scale (1.0 instead of 0.1 lowers $\|\mathbf{W}\|_2$ to 2.9 but the transient
growth stays at 15 and the run diverges on trial 3), a lower embedding order (2: diverges on trial 2).

**Solutions.** Two work, one of them is principled:

| ridge | $\|\mathbf{W}\|_2$ | max $\|W_{ij}\|$ | $\max_k \|(\mathbf{W}^\top)^k\|_2$ | reproduction error (max 0.4) | frozen value vs MS model | exact credit, ratio $5 \times 10^{-4}$ | local, CR trial 200 |
|---|---|---|---|---|---|---|---|
| $10^{-6}$ (old) | 18.5 | 9.4 | 71 | 0.0014 | 0.035 | diverges trial 2 | 5.15 |
| $10^{-4}$ | 6.7 | 4.5 | 11 | 0.008 | 0.014 | diverges trial 120 | 5.08 |
| $10^{-3}$ | 6.5 | 4.3 | 8.8 | 0.022 | 0.029 | stable, CR 5.06 | 5.07 |
| $3 \times 10^{-3}$ | 6.2 | 4.1 | 7.5 | 0.035 | 0.040 | stable, CR 5.10 | 5.07 |
| $10^{-2}$ | 5.4 | 3.6 | 5.4 | 0.060 | 0.079 | stable, CR 5.04 | 5.02 |
| $3 \times 10^{-2}$ | 4.3 | 3.1 | 4.4 | 0.092 | 0.136 | stable, CR 4.98 | 4.93 |
| $10^{-1}$ | 4.2 | 2.6 | 4.3 | 0.146 | 0.245 | stable, CR 4.81 | 4.77 |

1. **Regularise the fit** (`RNN_MS_RIDGE`). The ridge penalty shrinks the cancelling coefficients and is exactly
   the remedy for collinear regressors; it trades reproduction accuracy for conditioning (table). From $10^{-3}$
   the exact rule is stable at its default ratio and learns like the local rule (CR 5.06 against 5.07), while the
   block still reproduces the microstimuli to 2% of their maximum and the frozen network's value stays within 3% of
   the microstimulus model. The local rule is unaffected (CR 5.15 to 5.07). This is the fix adopted, see below.
2. **Bound the per-step sensitivity** (clip $\|\partial V / \partial \mathbf{W}\|_F$ to 1 or 0.1): stable, CR 3.8
   after 40 trials against 4.1 for the local rule. It is the standard remedy for exploding gradients in RNN
   training but it changes the plasticity rule for all models, and it is not biologically motivated.
3. **A balanced realisation** of the microstimulus impulse response (Ho-Kalman / balanced truncation) would give a
   near-normal $\mathbf{W}$ by construction, but its states are sign-indefinite and would need rectified pairs of
   units; the readout would then see linear mixtures of the microstimuli rather than the microstimuli themselves,
   so the frozen network would no longer learn like the microstimulus model. Not pursued.

**Robustness of the ridge fix** (exact credit, acquisition at ISI 25, 200 trials; "osc." = relative SD of the CR
level over the last 20 trials above 4%):

| ridge | ratio $5 \times 10^{-4}$ (default), seeds 0 to 4 | ratio $10^{-3}$, seed 0 | ratio $3 \times 10^{-3}$, seed 0 |
|---|---|---|---|
| $10^{-3}$ | seed 0 stable; seeds 1 and 2 diverge on trial 53 | stable | diverges trial 6 |
| $3 \times 10^{-3}$ | not tested | not tested | diverges trial 7 |
| $10^{-2}$ | all five stable (CR 5.04 to 5.07; seeds 1 and 2 osc., 4% and 7%) | stable | diverges trial 27 |
| $3 \times 10^{-2}$ | seed 1 diverges on trial 180, the others stable | not tested | not tested |

**Decision (2026-10-01, flagged):** `RNN_MS_RIDGE` is raised from $10^{-6}$ to $10^{-2}$ for both credit rules (one
block for both, so the comparison keeps a common initialization). It is the only value at which all five seeds are
stable at the default exact ratio; the margin is still thin (a six-fold larger ratio diverges, two seeds oscillate),
so the exact rule on this init remains marginal and is reported as such (in the five-seed scan above: one seed
diverges at ratio $10^{-3}$, three at $3 \times 10^{-3}$, and three of five seeds diverge in the timing set at
ISI 100, where probe trials leave the activity unperturbed for long stretches). Cost: the block now reproduces the
microstimuli to 0.06 (15% of their maximum) instead of 0.0014, and the frozen network's value deviates from the
microstimulus model by up to 8% instead of 3.5%; the local-rule result changes from CR 5.15 to 5.02. The
conservative alternative, keeping $10^{-6}$ and dropping the exact-credit microstimulus variant from the figures,
is a one-line revert.

## Notes for later work (2026-10-01)

Open items collected while finishing the RNN study, roughly by importance.

* **Offset pulses as an input of the recurrent network.** Without them the presence init cannot switch off on
  unreinforced probe trials, which drives its whole timing behaviour in the `rnn` study (plateau responses of up to
  107 steps with the local rule). Adding a CS offset pulse to $\mathbf{o}_t$ is a two-line change in
  `RecurrentNetwork.step` / `tasks`, but it is a modelling decision (the fixed presence representation has the
  offset for free, delta-TD deliberately does not).
* **Memory horizon of the random init.** With spectral radius 1.4 the activity of the random pool decays to 8% at
  100 steps, so the random init cannot learn ISI 50 (local) or 100 (exact). Options: scan the spectral radius (1.5
  explodes under rectification, so the usable window is narrow), add slow units (leaky integration $h_t = (1 - 1/\tau)
  h_{t-1} + \ldots$), or more units. Until then the blocking and overshadowing figures carry no information about the
  random init.
* **Stability of the recurrent plasticity.** There is no weight decay, normalisation or gradient control; divergence
  is detected after the fact (`DIVERGENCE_LIMIT`). The microstimulus init under exact credit (see the diagnosis
  above) is the clearest case where the plasticity rule, not the representation, is the problem. Candidates: a
  per-step bound on $\|\partial V / \partial \mathbf{W}\|$, a spectral-norm penalty, or synaptic scaling.
* **A trained random network can be unstable without the US.** In the reward-omission experiment of 2008 (999
  rewarded trials, then one omission), one of the five random-init seeds explodes on the omission trial itself: the
  value reaches $10^8$ within 100 steps after the expected reward, although all 999 rewarded trials were stable
  (CR 4.0) and the weights had moved by only $\|\mathbf{W} - \mathbf{W}_0\|_F = 0.23$. The learned recurrent
  weights have an expanding direction that the US-driven state normally interrupts; without the US the activity
  runs into it. The same networks survive the early-reward and omitted-second-cue probes, so the instability depends
  on how long the activity evolves unperturbed (the omission trial has 460 steps without input). This is the
  strongest argument for a stability mechanism in the plasticity rule (previous item), and it means that probe
  trials are not harmless for the plastic network: they test the dynamics far from the trained trajectory.
* **Cost of the exact credit assignment.** 1.9 ms per step against 0.17 ms for the local rule; the five-seed scan is
  dominated by it (about 12 of the 17 CPU hours). The backward chain stops early when the credit vector is all zero,
  but with the random init it rarely is. A truncated horizon of 50 steps would cut the cost threefold; whether it
  changes the results is untested (the step-size scan used 150).
* **Analyse the learned weights, not only the behaviour.** The dimensionality and the response width are indirect
  readouts. Storing $\mathbf{W}$ at the recorded trials would allow the eigenvalue spectrum, the growth of the
  presence self-connection towards $1/\gamma$, and whether the random init builds delay-line-like chains, to be
  shown directly.
* **Delta-TD on unreinforced trials at long ISIs.** The integrated value still explodes on probe trials at ISI 100
  (Fig. 4 and Fig. 6 reproductions, CR 140 and 40). The Q-formulation with action timing remains the proposed fix
  (see "The missing ingredient").
* **Seeds.** Five seeds are enough to show that the random-init conclusions are not a property of one weight matrix,
  but the SD of the step-size summary metrics near the stability boundary (trials to convergence, late fluctuation)
  will be large; more seeds for the scan alone would be cheap for the local rule.
* **Response rule and noisy values.** The random init's value fluctuates by about 0.05 within a trial; the leaky
  integrator with threshold smooths it, but the CR level of a probe trial is a maximum and therefore biased upwards
  by noise. A comparison of value-based and response-based metrics would show whether that matters.
