# Review of the 2M-interaction runs (REINFORCE / A2C / PPO)

Analysis of the executed notebooks at commit `787b3ea`. Nothing here is implemented —
this is diagnosis plus proposed fixes.

---

## Summary

| # | Symptom | Root cause | Is it a bug? |
|---|---|---|---|
| 1 | REINFORCE greedy return is always `0.00` progress / `-20.0` return | Not a bug. The learned **mean** throttle is ≈ 0. All of REINFORCE's forward motion comes from exploration noise being rectified by the `max(0.0, ...)` speed floor. With noise removed, `v` stays at exactly 0, which is an absorbing state. | No — real train/eval mismatch |
| 2 | A2C converges long before 2M | Correct observation, but the plateau is an artifact of the reward being nearly flat in lap time. Do **not** shrink the shared budget. | No |
| 3 | Car accelerates constantly | Correct. At 70 m/s through the seed-0 circuit's tightest corner the car pulls **33.6 g**. There is a pre-registered procedure for this decision in `EXPERIMENT.md`. | Known limitation |
| 4 | PPO takes ~2× the wall clock | Two independent causes, both measured: 320× more gradient steps per rollout (~14 min vs ~7 s), and its slow policy making greedy evaluation ~3× longer (~9 min vs ~3 min). | No |
| 5 | PPO peaks fast, then degrades and snakes | The learned exploration scale **grows** 0.61 → 1.9 (it does not collapse). Saturated tanh actions become bang-bang; the policy retreats to a slow, safe lap because the reward barely penalises being slow. | Design, not bug |
| 6 | — | Several. Biggest: every greedy evaluation runs **16 identical episodes** because `reset` is fully deterministic. | One real waste bug |

Three environment properties explain issues 1, 3 and 5 at once, so they are described first.

---

## The three shared environment facts

### F1 — `v = 0` is a fully absorbing state

In [src/envs/vehicle/kernel.py](src/envs/vehicle/kernel.py), `x`, `y` and `heading` all
advance proportionally to `speed`. At `v = 0` nothing moves, including heading: steering
has no effect. `reset` always places the car at the canonical start line with `speed=0.0`
([src/envs/racing/environment.py:94-124](src/envs/racing/environment.py#L94-L124)). So a
policy whose throttle is `≤ 0` at the start state is trapped: the observation never
changes, the action never changes, and the episode runs the full 1,000 steps.

Cost of that trap is exactly `1000 × time_penalty_rate × agent_timestep = 1000 × 0.5 × 0.04 = -20.0`.
That is precisely REINFORCE's final greedy return, to 12 decimal places, with maximum
progress exactly `0.0`.

### F2 — the speed floor rectifies exploration noise into free forward motion

`next_speed = min(max_speed, max(0.0, speed + dt·a))`. The clamp at zero is a **reflecting
barrier**. A throttle distribution with mean zero therefore produces a *positive* speed
drift, because negative excursions below zero are absorbed and positive ones are not.

Simulating pure zero-mean throttle (`tanh(N(0, 0.6065))`, the untrained policy) for 700
agent steps gives a mean speed of **2.36 m/s** with no learning at all.

This is why a policy can look like it is learning to drive while its throttle head has
learned nothing.

### F3 — the reward is nearly flat in lap time

Decomposition of a completed lap: `finish 100 + progress 100 - 0.02 per step`.

| lap | steps | return |
|---|---|---|
| fast lap (A2C) | 232 | 195.4 |
| slow lap (PPO) | 703 | 185.9 |

A car that is **3× slower** loses **4.9%** of its return. For a racing task the objective
barely mentions speed. This drives issue 2 (A2C's early plateau), issue 5 (PPO's retreat
to a slow safe lap), and it is why "converged" and "fast" are not the same thing here.

---

## 1. REINFORCE: greedy evaluation is always 0.00

**It is not an evaluation bug.** The path is correct: `normalizer.normalize(obs)` →
`agent.deterministic_action` → `tanh(mean(obs))`
([src/training/evaluation.py](src/training/evaluation.py),
[src/models/policies.py](src/models/policies.py)). The greedy policy genuinely does not move.

### Evidence that the throttle head never learned

The learned exploration scale is `σ = exp(-0.5) = 0.6065` at init. For a zero-mean
Gaussian squashed by `tanh`, `E|tanh(0.6065·ε)| = 0.4045`.

REINFORCE's plotted **mean throttle magnitude sits at 0.405 → 0.417 for the entire 3,200
episodes** — a perfectly flat line at the no-signal value. Inverting that magnitude for
the implied signed mean:

| run | plotted `E\|a\|` | implied mean `μ` | implied `E[a]` |
|---|---|---|---|
| REINFORCE | 0.417 | **0.045** | **0.034** |
| A2C | 0.72 | 1.10 | 0.70 |
| PPO (σ = 1.9) | 0.79 | ≈ 0.2 | ≈ 0.15 |

A2C learned a real throttle policy. REINFORCE's is statistically indistinguishable from
zero. And `0.034 × 9.26 m/s² × 21 s ≈ 6.6 m/s` of drift plus `≈ 2.4 m/s` of noise
rectification (F2) reproduces REINFORCE's observed mean speed of ~13 m/s almost exactly.

So: the training curve rising from −5 to +189 is REINFORCE learning **steering** (survive
longer → the rectified random walk accumulates more speed → higher progress). The throttle
was never learned. Remove the noise and F1 takes over.

The greedy return drops to exactly −20 at the *first* checkpoint (50k) and never recovers,
while training return keeps climbing for another 1.95M interactions.

### Two contributing causes

1. **`actor_weight_decay=1e-4`** is set only in the REINFORCE notebook. The actor weight
   norm decreases **monotonically and smoothly** across the whole run (11.68 → 11.40) — the
   decay term dominates the gradient term. Decay pulls the mean MLP output toward 0, i.e.
   toward `tanh(0) = 0` throttle, i.e. straight into F1. A2C and PPO have no weight decay,
   so this also breaks the Experiment 1 comparison.
2. **Only 200 optimizer updates in 2M interactions.** `completed_episodes_per_update=16`
   with ~625-step episodes gives 200 updates, versus 977 for A2C and 977 × 320 minibatch
   steps for PPO. Note the notebook is forced into 16 because
   [src/training/engines/reinforce.py](src/training/engines/reinforce.py) requires
   `environment_workers == collection_size`; the `ReinforceConfig` default is 8.

### Proposed fixes (in priority order)

- **Remove `actor_weight_decay` from the REINFORCE notebook**, or apply the same value to
  all three. Cheapest change; also required for a fair Experiment 1.
- **Break the absorbing start.** Either give `reset` a small non-zero initial speed
  (e.g. 2–5 m/s), or randomise the start: arc-length `s₀ ~ U(0, L)`, small lateral offset
  and heading jitter, small initial speed. This removes F1 from the greedy path entirely
  and improves state coverage for all three algorithms. It also fixes the 16-identical-
  evaluations waste in §6.
- **Add a no-progress termination**: end the episode (as a crash, or with truncation) if
  accumulated progress over the last ~2 s is below a small threshold. This deletes the
  1,000-step absorbing sink permanently instead of merely stepping around it.
- **Decouple `completed_episodes_per_update` from the worker count** so REINFORCE can use
  16 workers with a batch of 8 (400 updates) without changing the interaction budget.
- **Diagnostics:** the failure was invisible because the notebook plots `mean|throttle|` of
  the *behaviour* policy, whose noise floor (0.4045) masquerades as control. Plot the
  **signed** mean action and, separately, log the **greedy** action statistics and the
  greedy **outcome** (`completed` / `crashed` / `time_limit`). Had the outcome been plotted,
  "100% time_limit" would have been obvious at 50k interactions.

Do not expect the weight-decay removal alone to fix it — the absorbing start is the
structural cause, the decay just aims the policy at it.

---

## 2. Should the shared budget be smaller?

**No. Keep 2,000,000 for all three.** Reasons:

- `EXPERIMENT.md` already froze 2M as the Experiment 1 planning budget, with 40 evaluation
  positions, and defines right-censoring for runs that never converge. Sample-efficiency
  differences *are* one of the three reported outcomes ("Efficiency: actor size can change
  interactions and computation required"). Shrinking the budget to the fastest learner
  deletes the finding you are trying to measure.
- A2C's plateau is partly an illusion of the reward scale (F3). Its greedy return goes
  190 → 195 between 300k and 2M, which looks flat but is actually mean speed climbing
  20 → 30 m/s and lap time falling by ~35%. If you report **lap time** alongside return,
  A2C is visibly still improving at 2M.
- REINFORCE's problem is not the budget. At 2M it has learned nothing about throttle. A
  smaller budget hides that; a larger one would not fix it. Fix §1 first, then re-measure.

What I *would* change: the run is expensive (see §4 and §6.3), so cut cost from the parts
that produce no information — duplicate evaluations and PPO epochs — rather than from the
budget.

---

## 3. Physics: the car accelerates all the time

Your instinct is right, and the numbers are extreme. Seed-0 circuit: length 318 m, width
12 m, tightest corner radius **14.86 m**, 90th-percentile radius 24.9 m.

| speed | lateral acceleration at tightest corner |
|---|---|
| 30 m/s | 61 m/s² = **6.2 g** |
| 50 m/s | 168 m/s² = **17.1 g** |
| 70 m/s (A2C's final speed) | 330 m/s² = **33.6 g** |

The geometric minimum turn radius from `max_steering_angle=30°` and `wheelbase=3.6` is
6.24 m, implying a yaw rate of **11.2 rad/s (643 °/s)** at top speed. A2C's final greedy
rollout finishes at 69.5 m/s with throttle 0.88. `README.md:293` literally anticipates
this: *"If the car always uses full throttle → implement the planned grip-limited..."*.

**Proposal: do not hand-add physics now.** `EXPERIMENT.md` §"Version-0 physics decision"
already specifies a pre-registered trigger — train medium PPO on the *selected* Experiment 1
circuit with three dedicated roots, then add the grip model only if all four conditions hold
(≥100 visits to the top-curvature quartile, ≥5% with `v²|κ| > 4g`, median speed drop < 10%,
median throttle drop < 0.1). Everything needed is already recorded: `lateral_acceleration_proxy`
is on every retained trajectory row ([src/training/evaluation.py](src/training/evaluation.py)).
Run that procedure and let it decide — otherwise the physics choice becomes a post-hoc one
and the reported experiment loses its pre-registration.

My expectation is that the trigger will fire decisively.

Two caveats on the current notebook runs: they used `SINGLE_CIRCUIT_SEED = 0`, not the
circuit chosen by the 100-candidate selection rule, so they are development evidence only.
And when you do add the grip model, add the **smallest** one specified in `MDP.md`
(a `v²|κ| ≤ μg` speed/steer constraint) and keep version 0 selectable, as the protocol requires.

---

## 4. Why PPO takes ~2× as long

Measured on this machine (10 physical cores):

| measurement | value |
|---|---|
| single-process `RacingEnv.step` | **1.16 ms** (859 steps/s) |
| 10-worker vector step | 7.51 ms/wave → **1,332 steps/s** (1.55× a single process) |
| 16-worker vector step | 13.13 ms/wave → **1,218 steps/s** (*worse* — oversubscribed) |
| actor gradient step, batch 64 | **1.49 ms** |
| actor gradient step, batch 2048 | **3.70 ms** |

### Cause A — 320× more gradient steps per rollout (~14 min vs ~7 s)

PPO: 977 updates × 10 epochs × (2048 / 64) = **312,640 actor minibatch steps**, plus the
same for the critic. At 1.49 ms that is ≈ 466 s for the actor and ≈ 840 s (**14 min**)
including the critic.

A2C: 977 updates × 1 full-batch step = **977** steps at 3.70 ms ≈ **4 s**.

### Cause B — evaluation cost, which *is* caused by issue 5

41 checkpoints × 16 episodes = 656 greedy episodes per run, all single-process at 859 steps/s:

| algorithm | greedy episode length | eval steps | eval wall clock |
|---|---|---|---|
| A2C | ~232 | 152k | **~3 min** |
| PPO | ~703 | 461k | **~9 min** |
| REINFORCE | 1000 (stalled) | 656k | **~13 min** |

So PPO's slow policy (§5) directly inflates its own evaluation cost 3×. Your hunch that
issues 4 and 5 are related is correct — for this half of the cost.

### Rough totals

Collection is ~25 min for everyone (2M / 1,300 steps/s). Adding the above:
**A2C ≈ 28 min, PPO ≈ 48 min** — a 1.7× ratio, matching what you observed.

### Proposed fixes

- **`optimization_epochs: 10 → 4`.** Removes ~8 min and is also the right fix for §5.
  10 epochs on a 2048-row rollout is aggressive reuse for a task this noisy.
- **`EVALUATION_EPISODES: 16 → 1`** while the reset is deterministic — see §6.1. Saves
  ~8 min for PPO, ~12 min for REINFORCE, with **zero** information loss.
- **`environment_workers`: use 10, not 16.** 16 workers on 10 cores is measurably slower.
- Longer term, the env step itself is the real ceiling (§6.3).

---

## 5. PPO peaks, then degrades

### What actually happens

Greedy return: −12 → **195 within 50k interactions**, then repeated collapses (to ~0–15
around 350–450k, and dips to ~60–80 at 590k, 850k, 1.15M, 1.40M), settling at a stable but
**slower** 186 for the last 500k. Mean training speed: 2 → **31 m/s by episode 200** →
collapses → oscillates → settles at **11 m/s**.

### Cause 1 — the exploration scale grows, it does not collapse

This is the opposite of the usual PPO story, so it is worth stating plainly: `σ_throttle`
and `σ_steering` climb **0.61 → 1.9** over the run (the `log_std` bound is `(-5, 2)`, so
`σ` can reach 7.39 — it is nowhere near saturated). A2C's `σ` stays pinned at 0.60 and
REINFORCE's at 0.605.

At `σ = 1.9`, `E|tanh(u)| = 0.73` with a **zero** mean. PPO's plotted throttle magnitude
of 0.79 is therefore mostly saturation noise, not control. Every action is effectively
`±1` with a randomly flipping sign at 25 Hz — that is your "jumps left to right super
quickly".

The mechanism is a self-reinforcing loop specific to squashed Gaussians with no entropy
regularisation: saturated actions get positive advantage → the surrogate raises their
log-probability → the cheapest way to raise the log-probability of far-from-mean samples
is to raise `σ` → more saturation. `entropy_bonus_enabled=False`, so nothing opposes it,
and `kl_early_stop_enabled=False` and `value_clipping_enabled=False` remove the other two
brakes. Per-update KL (0.004–0.014) and clip fraction (0.03–0.20) look healthy — the drift
is slow and cumulative over 977 updates, not a single bad step.

**Note the counter-intuitive consequence: adding an entropy bonus would make this worse,
not better.**

### Cause 2 — the reward does not defend the fast policy (F3)

Once `σ` is large, the behaviour policy is erratic, so the mean action that maximises
advantage under that state distribution is *conservative*. And retreating from a 232-step
lap to a 703-step lap costs only 4.9% of return while removing most of the crash risk.
PPO is not malfunctioning — it is correctly optimising an objective that does not care
about lap time. A2C avoids this only because its `σ` never grew.

### Cause 3 — critic scale

`value_target_mean = 134`, critic weight norm grows unbounded 12 → **61**, critic loss
spikes to 157, and explained variance oscillates between 0.2 and 1.0 all run. `γ = 0.9995`
gives an effective horizon of 2,000 steps — longer than any episode — with no value
normalisation. The collapse events line up with the critic loss spikes.

### Proposed fixes

Ordered by expected effect:

1. **Cap the exploration scale.** Change `log_standard_deviation_bounds` from `(-5.0, 2.0)`
   to `(-5.0, 0.0)` (σ ≤ 1). One-line change in `ActorConfig`, directly targets the loop.
   As a diagnostic ablation, also try freezing `log_std` at init to confirm the mechanism.
2. **`optimization_epochs: 10 → 4`** and **enable `kl_early_stop_enabled`** with a target
   KL around 0.02. Also fixes half of §4.
3. **Enable `value_clipping_enabled`**, and normalise value targets (or standardise
   returns for the critic). Addresses the collapse events.
4. **Consider `γ = 0.999`** so the effective horizon (1,000 steps) matches the episode limit.
5. **Steering-rate limit in the vehicle model** — cap `|Δδ|` per agent step (e.g. 60 °/s,
   i.e. 2.4° per step). The greedy rollout snakes too, so the *mean* network is oscillating,
   not just the samples: this is high-gain feedback at 25 Hz with nothing damping it. An
   action-rate penalty in the reward is the alternative, but a rate limit is the honest
   physical fix and pairs naturally with the grip constraint in §3.

### "Should I stop the algorithm once it has learnt?"

No — and note `EXPERIMENT.md:317` already binds you: *"The final policy at exactly the
common budget is the primary result; a best-seen checkpoint is only a diagnostic."* Early
stopping on greedy performance would be selecting on the reported metric. Fix the
instability instead. If you want peak-vs-final as a *reported* quantity, add it as a
declared secondary metric ("final return / best-seen return") before freezing the manifest —
it is a genuinely interesting stability statistic across the three algorithms.

---

## 6. Other observations

### 6.1 Every greedy evaluation runs 16 identical episodes — this one is a real bug

`RacingEnv.reset` ignores its seed for state purposes: it always places the car at the
canonical start line with `speed=0.0`
([src/envs/racing/environment.py:94-124](src/envs/racing/environment.py#L94-L124)). The
policy is deterministic. The circuit is fixed. So all 16 evaluation episodes at every
checkpoint are **byte-identical**.

Confirmed in the data: `return_standard_deviation` and
`maximum_progress_standard_deviation` are **exactly 0.0 at every checkpoint in all three
notebooks**. The "mean ± one standard deviation" band on the greedy plots is not just
uninformative, it is misleading — it implies a spread was measured.

Cost: 15/16 of all evaluation compute. Across the planned Experiment 1 (45 runs × 41
checkpoints × 16 episodes) that is roughly **4.5 hours of pure duplication**.

Fix: either randomise the start state (preferred — it also fixes §1), or set
`EVALUATION_EPISODES = 1` until it is randomised. For Experiment 2, evaluate on N *distinct*
held-out circuits, which is where the ±1σ band becomes meaningful.

### 6.2 REINFORCE gets 200 updates; A2C 977; PPO 312,640

The interaction budget is a fair **data** budget, which is the right primary axis. But it
is not a fair **optimisation** budget, and the three differ by three orders of magnitude.
Report gradient steps and wall clock alongside interactions so a reader can see which
comparison a conclusion rests on. `EXPERIMENT.md:247` already asks for timing breakdowns —
add optimizer-step counts to that.

Also: REINFORCE ran with 16 workers and A2C/PPO with 10 (`ExecutionConfig` default =
physical cores). Since 16 is measurably *slower* than 10 on this machine, the wall-clock
comparison is currently confounded. Pin `environment_workers` explicitly in all three
notebooks.

### 6.3 The environment step is the real compute ceiling

1.16 ms per `RacingEnv.step` is slow for a 4-dimensional kinematic model. Profiling 3,000
steps:

- `project_to_segment` is called **165,018 times — 55 per agent step** — and accounts for
  33% of runtime. It is a scalar Python function called in a loop over ~11 candidate
  segments, 5 times per step (4 collision substeps + 1 observation).
- `scipy.interpolate.interp1d.__call__` accounts for a further ~25%, for heading/curvature
  lookups that could be `np.interp` on precomputed arrays.

Vectorising `_project_candidates` ([src/envs/geometry/projection.py:132](src/envs/geometry/projection.py#L132))
to evaluate all candidates in one NumPy call, and replacing the scipy interpolators, should
plausibly give a 3–5× step-rate improvement. Worth doing before Experiment 1: at the current
rate, 45 runs × 2M interactions is ~19 hours of collection alone, and the AsyncVectorEnv
only buys 1.55× because per-step IPC is comparable to the step itself.

### 6.4 The reward should be re-examined before the manifests are frozen

`EXPERIMENT.md` says *"The reward is unchanged throughout"* — so this is the moment to
decide, not later.

The two orderings from the previous fix must be preserved (they are guarded by
[tests/envs/test_reward.py](tests/envs/test_reward.py)):

- `crash_penalty < time_penalty_rate × agent_timestep × max_episode_steps` — otherwise
  standing still dominates driving.
- `progress_coefficient ≫ crash_penalty` — otherwise the return encodes *whether* you
  crashed, not *where*.

There is now a **third** ordering that is currently violated, and it is what makes a racing
agent indifferent to speed:

- **a fast lap must be worth materially more than a slow lap** — currently 195 vs 186.

Three options, with the arithmetic:

| option | effect on a 232-step vs 703-step lap | risk |
|---|---|---|
| `time_penalty_rate: 0.5 → 1.0` | 190 vs 172 (9.5% gap) | safe: a 50%-lap crash at 800 steps still scores +13 > −5 |
| `time_penalty_rate: 0.5 → 2.0` | 180 vs 144 (20% gap) | **unsafe**: a 50%-lap crash at 800 steps scores −19 < −5, so crashing immediately becomes preferable to driving badly — this re-inverts the ordering we just fixed |
| time-scaled finish bonus, e.g. `finish_reward × (2 − T_lap/T_max)` | 190 vs 170, tunable | safest: only applies on success, so it cannot make crashing attractive |

I would recommend the **third** option, optionally combined with a modest
`time_penalty_rate → 1.0`. Whichever you pick, add the new ordering as a fourth invariant
test next to the existing three, and freeze it before any reported run.

A cheaper alternative if you would rather not touch the reward at all: keep the reward and
report **lap time** as a primary outcome. That makes A2C's "plateau" visibly not a plateau
and PPO's degradation visibly a 3× regression rather than a 5% one.

### 6.5 The REINFORCE notebook's final cell was interrupted

Cell 24 of [notebooks/reinforce.ipynb](notebooks/reinforce.ipynb) ends in `KeyboardInterrupt`,
so there is no "after training" rollout line for REINFORCE (A2C and PPO both have one).
Re-run it after the §1 fixes.

### 6.6 Both critics grow without bound

A2C's critic weight norm goes 11.7 → 18.3 and PPO's 12 → 61, both still climbing at 2M with
no weight decay (`weight_decay_enabled=False`). Neither has diverged, but with value targets
around 80–134 and `γ = 0.9995` there is no mechanism holding them. Value-target normalisation
would address this and §5's collapse events together.

### 6.7 The circuit used is not the Experiment 1 circuit

All three runs used `SINGLE_CIRCUIT_SEED = 0` and `TRAIN_ON_RANDOM_CIRCUITS = False`. The
Experiment 1 circuit must come from the 100-candidate eligibility and median-centred
selection procedure and be saved to `tracks/experiment_1.json`. Worth doing that selection
before the physics decision in §3, since the grip trigger is defined on *that* circuit's
curvature quartiles.

---

## Suggested order of work before Experiment 1

1. **Randomise the start state** (arc-length, small lateral/heading jitter, small initial
   speed) — fixes §1's absorbing trap, §6.1's duplicate evaluations, and improves coverage
   for all three algorithms. Highest value per line changed.
2. **Remove `actor_weight_decay` from REINFORCE** (or apply uniformly) and add a no-progress
   termination.
3. **Fix the diagnostics**: signed mean action, greedy action statistics, greedy outcome
   distribution, lap time. Without these you cannot see whether step 1 worked.
4. **Cap `log_std` at `σ ≤ 1`, drop PPO to 4 epochs, enable KL early stop and value clipping.**
5. **Decide the reward's lap-time ordering** (§6.4) and freeze it, with a fourth invariant test.
6. **Select the Experiment 1 circuit**, then run the pre-registered version-0 physics
   decision (§3) and freeze the physics version.
7. **Optimise the env step** (§6.3) — optional, but it is ~19 hours of Experiment 1
   collection time.
8. Re-run the three notebooks and confirm: greedy and training curves agree, PPO's final
   equals its best within noise, and all three complete laps deterministically.

Steps 1–3 are the ones I would validate first, with a short 250k-interaction run per
algorithm. If REINFORCE's greedy progress leaves 0.00 in that run, the §1 diagnosis is
confirmed.
