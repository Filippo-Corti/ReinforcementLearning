# RL Car Racing

This repository contains all code and notes for the **RL Car Racing** project, for my Reinforcement Learning class.

## Project Overview (Project ID: PG-4):

**Main Focus**: 
Policy gradient, deep neural policies

**Scientific Objective**: 
Understand the impact of the complexity of the policy space on the performance of the agent and the amount of interactions needed to converge.

**Problem Description**: 
You want to control an autonomous Formula 1 car so that it completes a circuit in the shortest time possible. 
The car has access to its position (relative to the circuit) and velocity. 
It controls the acceleration and the steering wheel. It should be heavily penalized for going off track.

**Tasks**:
1. Choose/design a circuit (e.g. The Circuit de Monaco) and model the problem as an MDP with continuous actions. Think carefully about how to model the circuit and the relative position of the car. You are allowed to include in the state additional information about the specific circuit you are trying to solve, such as landmarks or information about the curvature, if you think this is helpful.
2. Define a reward function that encourages the agent to complete the circuit in the shortest time possible without going off track.
3. Define a parametric policy (Gaussian or deterministic) that maps states to actions (or mean actions) using a deep neural network (e.g. a fully connected neural network a.k.a. multi-layer perceptron). Implement it so that you can easily try neural networks of different sizes (number of layers and width of the layers).
4. Train your agent using a deep RL algorithm of your choice and compare the results obtained with policy networks of different sizes, in terms of:
    - Final performance
    - Number of training episodes needed to converge
    - Time needed to converge (this is machine dependent so make sure to run all experiments on the same computer)

**Challenging Variants**:
Try to learn a policy that can solve multiple circuits, in particular circuits not seen during training. You may want to train it on multiple, diverse circuits.


# Additional Notes

## Full Plan

0. **Task and MDP Formalization.**
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

## Policy parameterization

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

## Algorithms

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

## The scientific experiment (network size vs. performance/efficiency)

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

## Evaluation protocol

* Evaluate with the **deterministic** policy (use $\mu_\theta$, no sampling).
* Fixed set of eval seeds / start states, separate from training.
* Report a **task metric** (lap time, % of lap completed, off-track rate), not
  just raw return — return is reward-shaping-dependent and not comparable across
  reward tweaks.

## Final suggestions

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


## Variants and possible solutions

- If car goes always full throttle -> Add a lateral acceleration representing centripetal force. It is quadratically proportional to velocity, meaning that at high speed it is stronger. Maybe it should also influence steering effect? Or just move the car. Not sure yet.

- If training is unstable -> Add the previous action to the state observation

- If training is slow -> random starting distribution