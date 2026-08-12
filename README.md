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

The current implementation roadmap is maintained in [`PLAN.md`](PLAN.md). The
scientific comparisons, measures and analysis rules are specified separately in
[`docs/EXPERIMENT.md`](docs/EXPERIMENT.md). The exact policy, target and loss
equations are fixed in [`docs/LEARNING.md`](docs/LEARNING.md).

## Full Plan

0. **Task, track and MDP formalization.**
1. **Phase 1 — racing-environment baseline.** Build and validate the procedural
   track, Frenet geometry, kinematic dynamics, episode lifecycle and rendering.
2. **Phase 2 — experiment-ready learning system.** Implement deterministic
   training infrastructure, REINFORCE, A2C+GAE, PPO, grip-limited dynamics,
   LiDAR and multi-track orchestration. Run reduced-budget end-to-end validation
   on every experimental path.
3. **Experiment 1 — policy-space complexity.** Run REINFORCE, A2C+GAE and PPO
   with small, medium and large policy networks on one fixed circuit using
   Frenet observations and five paired training roots.
4. **Experiment 2 — generalization and observations.** Select a PPO policy size
   by the preregistered Experiment 1 rule, train on procedurally generated
   circuits, and compare Frenet with LiDAR on held-out circuits.
5. **Scientific report.** Analyze final task performance, interactions and time
   to convergence, computational cost, between-seed variation and generalization
   without hiding failed or non-converged runs.

The vehicle model is a kinematic bicycle constrained by a shared tyre friction
budget, a steering-rate limit and aerodynamic drag, so speed has to be spent and
recovered rather than simply held. The car is still a point for collision
detection and has no tyre slip or speed-dependent downforce. The full model,
including how excess cornering demand turns into understeer rather than a
special-cased failure, is specified in [`docs/MDP.md`](docs/MDP.md).

## View a track

Activate the repository's `.venv`, then open a deterministic generated track:

```bash
python experiments/view_track.py --seed 0
```

Or display a saved track definition:

```bash
python experiments/view_track.py --track tracks/example.json
```

Press Esc or close the Pygame window to exit. This viewer keeps the car at the
canonical start pose; interactive driving is a separate planned experiment.

## Drive a track manually

Launch the keyboard driver with a deterministic generated track:

```bash
python experiments/manual_drive.py --seed 0
```

Or drive a saved track definition:

```bash
python experiments/manual_drive.py --track tracks/example.json
```

Use W/S for throttle/brake, A/D to steer left/right, and R to reset the current
track. The window title shows speed, episode progress, reward and terminal
status. After a crash, completed lap or time limit, the final state remains on
screen until you reset or exit with Esc/window close.

## Validate the environment

Run the acceptance pass from a fresh repository virtual environment:

```bash
python -m venv /tmp/rl-phase1
/tmp/rl-phase1/bin/python -m pip install --editable .
/tmp/rl-phase1/bin/python experiments/phase1_acceptance.py --seed 0
```

The acceptance runner executes `pip check`, Black, Ruff, Pyright, bytecode
compilation, the full pytest suite and `git diff --check`. It also generates,
saves, reloads and RGB-renders a track; compares all outputs from two identical
action replays; runs Gymnasium's environment checker; and starts then quits the
manual driver through SDL's dummy video backend. Its pinned dependency set is
Python 3.12+ with Black 26.5.1, Gymnasium 1.3.0, Matplotlib 3.10, NumPy 2.4.4,
Pygame 2.6.1, Pyright 1.1.411, pytest 9.1.1, Ruff 0.16.0, SciPy 1.18.0 and setuptools
80.10.2 (see `requirements.txt` for the authoritative manifest).

The environment uses a point-car bicycle model with a tyre friction budget, a
steering-rate limit and drag. It has no tyre slip, load transfer, speed-dependent
downforce or finite vehicle footprint. Training samples the start pose around the
circuit while deterministic evaluation always launches from the canonical start
line. LiDAR and multi-circuit training remain Phase-2 work.

## Evaluate the reference controllers

The random-action and deterministic Frenet references are debugging baselines,
not training data. They generate and save one deterministic circuit, retain
JSONL episode/evaluation records, and keep their interactions separate from the
zero training counter:

```bash
python experiments/evaluate_references.py --seed 0 --run-path results/reduced_budget_end_to_end_validation/reference/seed_0
```

The selected run directory must be new or empty. It contains the versioned
`manifest.json`, `config.json`, `metadata.json`, episode/update/evaluation
JSONL streams, selected trajectories, checkpoints directory and an atomic
`completion.json`. Consumers reject a missing completion marker and reject
non-reported categories when reported run outputs are requested.

## Learning and experiment design

The three agents use one project-owned bounded Gaussian policy interface and a
shared MLP builder. Exact probability, return, advantage, bootstrapping and loss
semantics are frozen in [`docs/LEARNING.md`](docs/LEARNING.md). Course-lab agent
code is not reused by default.

Experiment 1 varies only the actor hidden sizes:

- small: `(32, 32)`;
- medium: `(64, 64)`; and
- large: `(256, 256)`.

For A2C and PPO, critic capacity remains fixed across actor sizes. Each of the
nine algorithm/size cells uses five paired training roots and an equal
environment-interaction budget. The primary evidence is lap completion, lap
time with its completion denominator, progress and crash rate. Return, learning-
curve area, interactions/time to convergence, throughput, memory and
optimization diagnostics are also retained.

Experiment 2 uses PPO and selects the smallest adequate actor through the rule
declared before seeing multi-track test results. Frenet and LiDAR runs are paired
by training root, procedural track schedule and held-out circuits. Evaluation is
deterministic and does not update normalization or consume training randomness.
See [`docs/EXPERIMENT.md`](docs/EXPERIMENT.md) for the complete protocol.

## Training engines

The `training.engines` package offers two deliberately different views of
training. `shared_engine.py` retains checkpointing, deterministic evaluation,
timing and experiment accounting for reproducible runs. The `reinforce.py`,
`a2c.py` and `ppo.py` modules contain direct algorithm-specific loops whose
collection and update boundaries can be read from top to bottom. These simpler
engines return episode and update records for notebooks and teaching; they do
not replace the shared engine used by `experiments/train.py`.

The readable engines also accept an episode-indexed procedural-track schedule.
When enabled, each episode reset generates the circuit associated with that
episode identity instead of selecting from a hand-picked in-memory pool. A
fixed root and episode index reproduce the same circuit independently of policy
sampling and environment-reset randomness.

[`notebooks/reinforce.ipynb`](notebooks/reinforce.ipynb) walks through a
REINFORCE run with the readable engine. Its configuration exposes one exact
training-interaction budget and a switch between a fixed reference circuit and
a fresh episode-indexed procedural training circuit. The live viewer accepts a
step count: the initial policy shows 20 actions, while the final policy can use
the full 1000-step episode. Inline mode places the circuit beside current
progress, return, speed, throttle and steering plus speed/control histories;
window mode synchronizes the Pygame view with a Matplotlib statistics window.

[`notebooks/a2c.ipynb`](notebooks/a2c.ipynb) and
[`notebooks/ppo.ipynb`](notebooks/ppo.ipynb) present the corresponding readable
actor-critic engines. All three use the same section order and categorized
figures for task performance, progress/episode efficiency, driving behaviour,
and optimization/exploration. Common plots include moving-average training
return, greedy return with a standard-deviation band, training and greedy
maximum progress, episode length, actor loss, actor/critic weight norms,
learned throttle/steering sigma, mean speed and mean throttle magnitude. A2C
retains GAE and critic-fit diagnostics; PPO additionally shows clipping and
approximate KL. Persistent CPU workers collect one synchronous observation
batch at a time; A2C and PPO retain an explicit time-by-environment rollout so
GAE remains independent between workers.

## Train an agent

The shared training entry point exposes REINFORCE, A2C and PPO. Learning rates
are intentionally required: they must come from the documented pre-experiment
calibration procedure rather than an implicit script default. The default
worker count is eight for REINFORCE and the number of physical CPU cores for A2C
and PPO; `--num-envs` makes it explicit. For a saved circuit, run:

```bash
python experiments/train.py --algorithm reinforce --seed 0 --track tracks/experiment_1.json --run-path results/reduced_budget_end_to_end_validation/reinforce/seed_0 --actor-size medium --actor-learning-rate 0.0003 --interaction-budget 2000000
```

The run directory records complete training episodes, REINFORCE updates,
deterministic evaluations when enabled, and a final checkpoint. A partial final
eight-episode batch remains counted in the interaction budget but is not used
for an optimizer update, as required by the Monte Carlo target.

## Analyze recorded results

Regenerate Experiment 1 tables and figures only from complete reported run
directories:

```bash
python experiments/analyze_results.py --experiment 1 --results-root results/reported_experiments/experiment_1 --output results/reported_experiments/experiment_1_analysis
```

Use `--experiment 2` for the observation-generalization study. Its optional
`--geometry-specification` JSON supplies the frozen length and curvature bin
edges; the analysis command never invents those experimental thresholds.

The output contains canonical JSON and CSV tables, figures, and an
`analysis_manifest.json` that identifies every input run and checksum. Runs are
aggregated by training root, paired by recorded root/circuit identity, and
processed in canonical order. Re-running the command produces byte-identical
data tables. Reported training commands must also provide
`--near-saturated-steering-threshold`; this protocol value has no implicit
default.

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
  Train on diverse seeds from the frozen procedural generator, hold out disjoint
  circuit seeds, and report the **train/test gap**.
  This is where the Frenet+curvature-preview observation shines: it's
  track-agnostic by construction, whereas a policy that memorizes absolute
  $(x,y)$ cannot generalize. Worth stating explicitly as a hypothesis and testing
  it — it's a clean, gradeable scientific result.

* **Keep training and evaluation starts distinct.** Training samples a pose
  around the circuit for state coverage; deterministic evaluation always launches
  from the canonical start line so every reported number answers the same
  question. Mixing the two makes evaluation curves incomparable.

* **Keep the frozen dynamics.** The friction budget, steering-rate limit and drag
  are specified in `docs/MDP.md` and are the physics for both reported
  experiments. Tyre slip, load transfer and a finite vehicle footprint remain
  future work.

* **Metrics/tracking.** Even without `wandb`/`tensorboard` installed, a small CSV
  logger per run + the planned project plotting utilities is enough. What
  matters for the grade is *comparable* curves across net sizes on one machine —
  decide the metric, the x-axis (env steps), and the seed count **before** you
  start the sweep, so you don't re-run everything.


## Variants and possible solutions

- If the car always uses full throttle -> check the friction budget is actually
  binding before retuning the reward; on this circuit it caps the tightest corner
  at about 17 m/s.

- If training is unstable -> diagnose the frozen configuration; changing the
  observation requires a new protocol amendment

- If training is slow -> check the outcome mix and the *signed* mean throttle
  before the return. A flat throttle magnitude with a zero signed mean means the
  policy has learned nothing and the motion is exploration noise.

- Infinite horizon -> The car is supposed to keep going indefinitely -> I would have to make sure that it does not crash intentionally
