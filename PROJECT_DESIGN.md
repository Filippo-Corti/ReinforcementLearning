# RL Car Racing — Design Review & Plan

*A companion to the proposal in [README.md](README.md). This document reviews the
MDP formalization, proposes a way to complete it, lays out a design/solution
roadmap (REINFORCE → Actor–Critic → PPO), sketches the code structure, and
collects a set of practical notes that don't fit elsewhere.*

---

## 1. On the MDP Formalization

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
