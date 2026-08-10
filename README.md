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
   training infrastructure, REINFORCE, A2C+GAE, PPO, conditional lateral grip,
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

The version 0 kinematic model intentionally has no lateral grip limit, drag,
finite vehicle footprint, tire slip, or steering-rate limit. Full throttle may
therefore remain optimal in corners. A capable learned policy will first be
diagnosed on version 0. The minimum grip constraint will be added before the
reported experiment runs only if the pre-experiment capability trigger is met;
the reward remains unchanged during that decision.

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
Python 3.12+ with Black 26.5.1, Gymnasium 1.3.0, NumPy 2.4.4, Pygame 2.6.1,
Pyright 1.1.411, pytest 9.1.1, Ruff 0.16.0, SciPy 1.18.0 and setuptools
80.10.2 (see `requirements.txt` for the authoritative manifest).

Version 0 deliberately uses a point-car kinematic bicycle model. It has no
lateral-grip limit, aerodynamic drag, tire slip, steering-rate limit or finite
vehicle footprint, so it can make unrealistic full-throttle cornering possible.
Learning agents, LiDAR and multi-circuit training are Phase-2 work; randomized
starts and more detailed vehicle dynamics remain outside the approved
experiments.

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

## Train REINFORCE

The shared training entry point currently exposes the actor-only REINFORCE
learner. Its learning rate is intentionally required: it must come from the
documented pre-experiment calibration procedure rather than an implicit script
default. For a saved circuit, run:

```bash
python experiments/train.py --algorithm reinforce --seed 0 --track tracks/experiment_1.json --run-path results/reduced_budget_end_to_end_validation/reinforce/seed_0 --actor-size medium --actor-learning-rate 0.0003 --interaction-budget 2000000
```

The run directory records complete training episodes, REINFORCE updates,
deterministic evaluations when enabled, and a final checkpoint. A partial final
eight-episode batch remains counted in the interaction budget but is not used
for an optimizer update, as required by the Monte Carlo target.

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

* **Keep the approved initial-state rule fixed.** The experiments use the
  canonical start. Randomized starts are a possible later study, not a training
  shortcut for selected reported experiment runs.

* **Keep version-0 dynamics until the registered diagnosis.** The current
  kinematic bicycle has no friction limit. Add only the minimum grip constraint
  if the capable-policy trigger in the protocol is met; richer tire and vehicle
  dynamics remain future work.

* **Metrics/tracking.** Even without `wandb`/`tensorboard` installed, a small CSV
  logger per run + the planned project plotting utilities is enough. What
  matters for the grade is *comparable* curves across net sizes on one machine —
  decide the metric, the x-axis (env steps), and the seed count **before** you
  start the sweep, so you don't re-run everything.


## Variants and possible solutions

- If the car always uses full throttle -> implement the planned grip-limited
  dynamics before retuning the reward.

- If training is unstable -> diagnose the frozen configuration; changing the
  observation requires a new protocol amendment

- If training is slow -> retain the fixed start for the approved reported
  experiments; a random-start curriculum is a separate variant

- Infinite horizon -> The car is supposed to keep going indefinitely -> I would have to make sure that it does not crash intentionally
