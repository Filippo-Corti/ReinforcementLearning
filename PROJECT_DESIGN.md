# RL Car Racing — Design Review & Plan

*A companion to the proposal in [README.md](README.md). This document reviews the
MDP formalization, proposes a way to complete it, lays out a design/solution
roadmap (REINFORCE → Actor–Critic → PPO), sketches the code structure, and
collects a set of practical notes that don't fit elsewhere.*

---

## 1. On the MDP Formalization

The skeleton is solid: separating the **true environment state** from **two
alternative observation models** (Frenet vs. LiDAR) is exactly the right framing
for the scientific objective, because it lets you study how the *representation*
(and therefore the policy space) interacts with sample efficiency. Below I go
component by component with concerns, then give a completed formalization.

### 1.1 Environment state — concerns

$$ s_t = (x_t, y_t, \theta_t, v_t, C) $$

* **`C` is a parameter, not a state.** The circuit configuration never changes
  within an episode, so strictly it belongs to the *MDP definition*, not to the
  time-varying state. It only becomes a random variable when you attack the
  multi-circuit variant, where each episode samples $C \sim \mathcal{D}$. I'd
  keep it in the tuple for clarity but explicitly note it is constant along a
  trajectory (`\dot C = 0`).

* **Missing progress variable.** You cannot detect lap completion, compute a
  progress reward, or even index the local curvature without the car's
  **arc-length position** $s_t$ along the centerline. This is the single most
  important omission. Add $s_t$ (and a lap counter if you want multi-lap).

* **The kinematic model has no notion of grip.** With scalar speed $v$ and no
  lateral dynamics, *nothing stops the car from taking any curve at any speed*.
  That quietly destroys the "minimum-time" problem: the optimal policy becomes
  "full throttle, always," and there is never a reason to brake. See §1.4 — this
  needs a fix at the dynamics level, not the reward level.

### 1.2 Observation (1), Frenet — concerns

$$ o_t = (d_t, \phi_{e,t}, v_t) $$

* **This is not a Markov (sufficient) statistic, so the "Fully Observable"
  label is not yet earned.** Given only $(d, \phi_e, v)$ the agent knows where it
  sits *relative to the track right here*, but not what is *coming*. Two points
  on the circuit with identical $(d,\phi_e,v)$ but different upcoming curvature
  demand different actions (brake vs. accelerate), so no stationary policy on
  this observation can be optimal. You've dropped the dependence on $s_t$, which
  is precisely what carries the curvature information.

* **Fix — add a curvature preview.** Include the signed centerline curvature at a
  set of look-ahead distances,
  $\kappa(s_t + \ell_1), \dots, \kappa(s_t + \ell_K)$ for a fixed horizon of
  preview points $\{\ell_k\}$. This is the "landmarks / curvature" information
  the proposal (Task 1) explicitly permits, and it is what makes the Frenet
  observation an honest fully-observed MDP. It also mirrors how real racing-line
  controllers and F1-style RL agents are fed the track ahead.

* Optionally decompose speed into longitudinal/lateral components if you ever
  move to a dynamic (non-kinematic) model; with the kinematic bicycle, scalar
  $v$ is fine.

### 1.3 Observation (2), LiDAR — concerns

$$ o_t = (v_t, R) $$

* Correctly identified as a genuine **POMDP** — good. Two caveats worth stating:
  * A **single** scan doesn't encode *direction of progress*: the geometry looks
    similar whether the car faces "up-track" or "down-track." Either the reward
    must make backwards motion clearly bad, or you should give the agent a way to
    tell — e.g. **frame stacking** (last $k$ scans) or including the previous
    action, both of which also restore approximate Markovianity of the velocity.
  * Specify the sensor concretely: number of beams $N_R$, angular field of view
    (e.g. a $180^\circ$ forward fan), maximum range $R_{\max}$, and the
    normalization ($R/R_{\max} \in [0,1]$). These are hyperparameters that
    materially affect the policy space you're studying.

### 1.4 Action space & transition — concerns

Action space $a_t \in [-1,1]^2$ and the bicycle model are appropriate (this is
the standard **rear-axle kinematic bicycle**, and your sign conventions are
internally consistent: $\delta>0 \Rightarrow \dot\theta>0 \Rightarrow$ left).
Two real gaps:

* **Unclamped speed.** "No reversing" must be enforced as
  $v_{t+1} = \mathrm{clip}(v_t + \Delta_t\,\bar a^{throttle}_t,\; 0,\; v_{\max})$.
  Without the lower clamp, braking at rest gives negative speed; without an upper
  clamp, speed diverges. Pick a physically motivated $v_{\max}$.

* **No friction / grip limit (the important one).** Add a lateral-acceleration
  constraint so that cornering *costs* speed. The lateral acceleration under the
  kinematic model is $a_{lat} = v\,\dot\theta = v^2 \tan(\bar a^{steer})/L$.
  Impose a friction limit $|a_{lat}| \le \mu g$ and, when it is exceeded, either
  (a) terminate with an "off-track / spin" penalty, or (b) saturate the achievable
  yaw rate. This is what turns "go fast" into a genuine speed-vs-curvature
  trade-off and makes the scientific question about policy complexity
  meaningful. A light longitudinal drag term $-c_d v$ is a cheap, optional extra.

* **Integration.** Forward Euler is fine but gets inaccurate at large $\Delta_t$
  and high speed. Use a small physics step (e.g. sub-step the integrator a few
  times per control step, or move to semi-implicit/RK2). State $\Delta_t$
  explicitly (e.g. control at 20–50 Hz).

* **Noise.** The additive-Gaussian option is reasonable; just don't perturb $C$,
  and re-apply the $v\ge 0$ clamp after adding noise.

### 1.5 What's still undefined

To be a complete MDP you still owe four objects:

* **Initial-state distribution** $\rho_0$: e.g. start on the centerline at
  $s_0=0$ with $\phi_e=0$, $v_0=0$ (or randomize $s_0$, small $d_0,\phi_{e,0}$,
  and $v_0$ for a richer start distribution — this helps exploration a lot).
* **Termination vs. truncation** (Gymnasium distinguishes these):
  * *terminated* — off track ($|d_t| > w(s_t)/2$), grip exceeded (if you chose
    hard termination), or lap completed ($s_t \ge L_{\text{track}}$);
  * *truncated* — episode step budget $T_{\max}$ reached.
* **Discount** $\gamma$ (e.g. $0.99$) and the horizon $T_{\max}$.
* **Track geometry** $C$: an explicit centerline (arc-length parameterized) plus
  a (possibly varying) half-width $w(s)$ defining the drivable corridor.

### 1.6 A completed reward function

The README leaves this empty. The cleanest **minimum-time** shaping is a **dense
progress reward** — reward arc-length gained per step — which, at fixed control
rate, is equivalent to maximizing average speed:

$$
r_t \;=\; \underbrace{\lambda_{\text{prog}}\,(s_{t+1}-s_t)}_{\text{progress}}
\;-\; \underbrace{\lambda_{\text{time}}}_{\text{per-step cost}}
\;-\; \underbrace{\lambda_{\text{ctrl}}\,\lVert a_t - a_{t-1}\rVert^2}_{\text{smoothness (optional)}}
\;+\; \underbrace{R_{\text{lap}}\,\mathbb{1}[\text{lap complete}]}_{\text{terminal bonus}}
\;-\; \underbrace{R_{\text{off}}\,\mathbb{1}[\text{off track / spin}]}_{\text{terminal penalty}}
$$

Design notes:

* **Progress $s_{t+1}-s_t$** is the workhorse; it's dense, well-shaped, and
  directly proportional to speed. Use the *signed* projection onto the centerline
  so backwards motion is penalized automatically.
* Keep either the per-step time cost **or** rely on progress-in-fixed-horizon,
  not both aggressively — they encode the same objective and double-tuning them
  is a common source of pathologies.
* Beware **reward hacking**: agents love to (i) sit still if off-track penalty
  ≫ progress reward is mis-scaled, (ii) cut corners across the inside if $d$ is
  unbounded, (iii) oscillate across the centerline to farm projection error. The
  smoothness term and a correct off-track termination address most of these.
* Make $R_{\text{off}}$ a **terminal** penalty (episode ends), not a per-step
  one — otherwise the agent may learn to "camp" just outside to minimize its
  accumulated cost.

### 1.7 Completed formalization (summary)

| Object | Definition |
|---|---|
| True state | $s_t=(x_t,y_t,\theta_t,v_t,s_t^{arc};\,C)$, with $C$ constant per episode |
| Frenet obs. | $o_t=(d_t,\phi_{e,t},v_t,\ \kappa(s_t{+}\ell_1),\dots,\kappa(s_t{+}\ell_K))$ — Markov |
| LiDAR obs. | $o_t=(v_t, R_t,\ [\,\text{prev. }k\text{ frames / last action}\,])$ — POMDP |
| Action | $a_t=(a^{throttle}_t,a^{steer}_t)\in[-1,1]^2$, mapped as in README |
| Dynamics | rear-axle kinematic bicycle + $v\!\in\![0,v_{\max}]$ clamp + friction limit $|v^2\tan\delta/L|\le\mu g$, sub-stepped Euler, optional Gaussian noise |
| Reward | dense progress − time − control + lap bonus − off-track (terminal) |
| $\rho_0$ | centerline start (optionally randomized $s_0,d_0,\phi_{e,0},v_0$) |
| Termination | off-track / grip-exceeded / lap done; truncation at $T_{\max}$ |
| Discount | $\gamma\approx0.99$ |

---

## 2. How I'd Design the Problem & the Solution

### 2.1 Staging (de-risk in this order)

1. **Frenet + kinematic env, single circuit.** Fully-observed, smooth, cheap.
   Get REINFORCE to *move forward at all* before touching anything fancy.
2. **Add the grip limit and tune reward scales** until the learned policy
   visibly brakes for corners. This is your "the environment is actually
   interesting" checkpoint.
3. **Switch algorithms** REINFORCE → A2C → PPO on the *same* env, so algorithmic
   gains aren't confounded with env changes.
4. **Run the network-size sweep** (the graded scientific objective) with your
   best-performing algorithm.
5. **Only then** tackle the LiDAR POMDP and the multi-circuit generalization
   variant.

### 2.2 Policy parameterization

* **Gaussian policy** $\pi_\theta(a|o)=\mathcal{N}(\mu_\theta(o),\,\sigma^2)$ with
  an MLP mean. Start with a **state-independent, learnable log-std** (a single
  parameter vector) — it's stabler than a state-dependent head early on.
* **Bound the actions correctly.** Either squash with $\tanh$ (and use the
  change-of-variables correction in $\log\pi$ if you go the SAC-style route) or,
  simpler for PPO, keep an unbounded Gaussian and **clip** the sampled action to
  $[-1,1]$ in the env. Pick one and be consistent; the $\tanh$ correction is a
  classic silent bug.
* Make **depth and width first-class knobs** (e.g. `hidden_sizes=(64,64)`), since
  varying them *is* the experiment. Keep everything else fixed across the sweep.

### 2.3 Algorithms

* **REINFORCE (baseline, already implemented in lab 05).** Reuse your batched
  `collect_batch` / `reinforce` machinery. Add a **baseline** (a value network or
  even a batch-mean return) to cut variance — this is the natural bridge to
  Actor–Critic and costs almost nothing.

* **Actor–Critic / A2C (next).** Add a critic $V_\phi(o)$ and switch the policy
  gradient to use an **advantage** estimate. Use **GAE($\lambda$)** for the
  advantage — it's a small function and pays for itself immediately. Train actor
  and critic together; normalize advantages per batch.

* **PPO (the workhorse).** Clipped surrogate objective, multiple epochs of
  minibatch updates over each rollout, value-clipping optional, an **entropy
  bonus** to keep exploration alive, and (optionally) an early-stop on a KL
  threshold. PPO is the right final algorithm for continuous control at this
  scale and is what your sweep results should be reported on.

  Practical settings that matter more than the algorithm choice:
  * **Observation normalization** (running mean/std) — near-mandatory for PPO.
  * **Reward/return scaling** or advantage normalization.
  * **Vectorized environments** for throughput (Gymnasium `SyncVectorEnv` /
    `AsyncVectorEnv`).
  * Orthogonal init, `tanh` activations, Adam with a modest LR (~3e-4), GAE
    $\lambda\approx0.95$.

### 2.4 The scientific experiment (network size vs. performance/efficiency)

Hold the algorithm (PPO), env, reward, and all other hyperparameters fixed. Vary
only `hidden_sizes` across, say, `{(32,), (64,64), (256,256), (256,256,256)}`.
For each configuration, run **≥3–5 seeds** and report:

* **Final performance** — mean ± std of *deterministic* (mean-action) evaluation
  return / best lap time, over held-out eval episodes.
* **Sample efficiency** — environment steps (not iterations) to reach a fixed
  return threshold.
* **Wall-clock to converge** — on one fixed machine, as the proposal requires;
  log it alongside step counts so you can separate "needs more data" from "needs
  more compute per step."

Plot learning curves with seed variance bands (mean ± std or IQM). The expected
narrative — bigger nets → better asymptote but more data/compute, with
diminishing or negative returns past some width — is exactly what the grader is
looking for; make the plots say it.

### 2.5 Evaluation protocol (decide this up front)

* Evaluate with the **deterministic** policy (use $\mu_\theta$, no sampling).
* Fixed set of eval seeds / start states, separate from training.
* Report a **task metric** (lap time, % of lap completed, off-track rate), not
  just raw return — return is reward-shaping-dependent and not comparable across
  reward tweaks.

---

## 3. How I'd Structure the Code

Extend the existing course package `rlc/` rather than starting fresh — it already
has `agents/`, `envs/`, `utils/`, a Gymnasium-based env convention, and a clean
docstring style. Suggested layout:

```
rlc/
├── envs/
│   ├── racing.py          # RacingEnv(gym.Env): dynamics, reward, termination, render
│   ├── track.py           # Track geometry: centerline spline, width w(s),
│   │                      #   arc-length <-> (x,y) (Frenet) conversion, curvature κ(s)
│   ├── observations.py    # FrenetObservation / LidarObservation wrappers or builders
│   └── lidar.py           # ray-casting against the track boundary
├── models/                # (new) reusable neural components
│   ├── mlp.py             # make_mlp(in, out, hidden_sizes, activation) — the size knob
│   ├── policies.py        # GaussianPolicy (mean MLP + log-std), deterministic policy
│   └── value.py           # ValueNetwork critic
├── agents/
│   ├── reinforce.py       # (from lab 05) + optional baseline
│   ├── actor_critic.py    # A2C with GAE
│   └── ppo.py             # clipped PPO
├── utils/
│   ├── buffers.py         # RolloutBuffer, GAE computation
│   ├── normalizers.py     # RunningMeanStd for obs / returns
│   ├── training.py        # (extend existing) generic train/eval loops
│   ├── seeding.py         # SeedSequence spawning (mirror lab 05's discipline)
│   └── plotting.py        # (extend existing) learning-curve + track-trajectory plots
├── configs/               # dataclass or YAML experiment configs (net size sweep, etc.)
└── tracks/                # circuit definitions (e.g. monaco.json / oval.json)

experiments/               # thin runner scripts / notebooks that call into rlc/
tests/                     # env + geometry unit tests
```

Design principles to keep it clean and reproducible:

* **Env owns physics + reward; agents own learning.** The env should be
  swappable between Frenet and LiDAR observation without touching agent code —
  do it via a Gymnasium `ObservationWrapper` or an `observation="frenet"|"lidar"`
  constructor flag, so both share one dynamics core.
* **One shared MLP builder** parameterized by `hidden_sizes` is what makes the
  network-size sweep a one-line config change instead of copy-paste.
* **Preserve the lab's agent contract** (`select_action` / `update` /
  `end_episode`, or the batched `collect_batch` pattern) so your existing
  training utilities and plotting keep working across all three algorithms.
* **Track as data, not code.** Represent circuits as arc-length-sampled
  centerlines + widths in `tracks/*.json`; build a KD-tree (or nearest-segment
  search) once for the point→Frenet projection. This makes the multi-circuit
  variant "load a different file," and lets you procedurally generate tracks for
  domain randomization.
* **Config objects + explicit seeds** for every run (reuse the `SeedSequence`
  discipline from lab 05 — seed env and policy independently, which you already
  do). Log hyperparameters with results so experiments are reconstructible.
* **Rendering** with `pygame` (already installed) as an `rgb_array` render mode,
  matching the course's env convention — invaluable for debugging reward hacking
  by *watching* the car.

---

## 4. Additional Notes

* **Track/Frenet math is where the bugs will live.** The point-to-centerline
  projection (needed for $d$, $\phi_e$, and $s$) is the most error-prone piece.
  Write a **round-trip unit test**: `frenet(xy_from_frenet(s, d)) ≈ (s, d)`.
  Also test that curvature $\kappa(s)$ has the right sign on a known arc, and run
  Gymnasium's `check_env` on `RacingEnv`. Cheap tests here save days.

* **Debugging RL is mostly debugging the env and the reward.** Before blaming
  the algorithm, sanity-check: does a hand-coded "aim at the centerline, throttle
  proportional to how straight the road ahead is" controller get a sensible
  return? If a scripted policy can't, the reward/termination is wrong, not the
  net. Keep such a **scripted baseline** around as a reference line on every
  plot.

* **Watch for classic PPO failure modes:** entropy collapsing to a deterministic
  policy too early (raise the entropy coefficient or the initial log-std), value
  loss exploding (normalize returns / clip value), and KL spikes (lower LR or
  clip range). Log entropy, approx-KL, clip fraction, and explained variance of
  the value function every iteration — these four diagnostics tell you almost
  everything.

* **The multi-circuit variant is a generalization study — treat it as such.**
  Train on a diverse *set* of tracks (procedural generation + curvature/width
  randomization), hold out several tracks, and report the **train/test gap**.
  This is where the Frenet+curvature-preview observation shines: it's
  track-agnostic by construction, whereas a policy that memorizes absolute
  $(x,y)$ cannot generalize. Worth stating explicitly as a hypothesis and testing
  it — it's a clean, gradeable scientific result.

* **Curriculum helps.** Randomizing the start position along the track (§1.5) and
  optionally starting near hard corners dramatically speeds early learning versus
  always starting from a standstill at $s_0=0$.

* **Keep the dynamics kinematic unless you have time to spare.** The kinematic
  bicycle + friction limit already gives a real speed/curvature trade-off. A full
  dynamic model (tire slip, load transfer) is more realistic but adds state,
  tuning, and instability for little pedagogical gain relative to the stated
  objective. Note it as future work.

* **Metrics/tracking.** Even without `wandb`/`tensorboard` installed, a small CSV
  logger per run + your existing `plotting.py` is enough. What matters for the
  grade is *comparable* curves across net sizes on one machine — decide the
  metric, the x-axis (env steps), and the seed count **before** you start the
  sweep, so you don't re-run everything.
