# AGENTS.md — Project Context for AI-Assisted Development

## Project overview

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

## General directions
- Check out the `PLAN.md` file for the currently approved execution roadmap and follow its steps in order.
- Check out the `docs/MDP.md` file for a formalization of the underlying MDP.
- Check out the `docs/TRACK.md` file for info on how the racing track should be generated, saved into memory and used.
- Check out the `docs/DIARY.md` file everytime you need to track the progress you have made on the project.

## Workflow rules
1. **Plan before code.** For anything nontrivial, outline the approach, build a plan, and list files to be touched. Wait for confirmation before writing any code.
2. **One concern per task/commit.** Don't mix algorithm changes with logging/refactor changes in the same diff. Once you have finished your task, always commit to branch main.
3. **Tests accompany code.** Any new function gets at least one test. Any behavior change gets a regression test that would have caught the previous behavior.
4. **No silent hyperparameter choices.** If a value isn't specified anywhere, ask or flag it as a placeholder — don't pick a "reasonable default" silently in core files.
5. **Determinism.** All training runs must accept a `--seed` and be reproducible given the same seed + config.
6. **Cite, don't recall.** For anything algorithm-specific, work from the equations/spec provided in this repo or pasted into the prompt rather than from memory of "how PPO usually works" — implementations vary across papers/codebases.
7. **Document your changes.** Every functionality that is added/modified/removed to the project should be properly documented not only inside the code and with the git commits, but also in the `docs/DIARY.md` diary file. You can find examples of how to document changes there. Report the commit from point 2 in the diary.
8. **Do not reuse course-lab code by default.** Code on the `lectures` branch, in lab notebooks, or in earlier exercises is reference material only. Implement project-owned agents and utilities from scratch unless the user explicitly asks to migrate or reuse a specific piece.

## Suggested code structure:
This is the suggested layout. Feel free to adapt it as you wish, but make sure to update this if you ever choose to:

```
src/
├── envs/
│   ├── racing.py          # RacingEnv(gym.Env): dynamics, reward, termination, render
│   ├── track.py           # Track geometry: centerline spline, width w(s),
│   │                      #   arc-length <-> (x,y) (Frenet) conversion, curvature κ(s)
│   ├── observations.py    # FrenetObservation / LidarObservation wrappers or builders
│   └── lidar.py           # ray-casting against the track boundary
├── models/                # reusable neural components
│   ├── mlp.py             # make_mlp(in, out, hidden_sizes, activation) — the size knob
│   ├── policies.py        # GaussianPolicy (mean MLP + log-std), deterministic policy
│   └── value.py           # ValueNetwork critic
├── agents/
│   ├── reinforce.py       # project-owned REINFORCE + optional baseline
│   ├── actor_critic.py    # A2C with GAE
│   └── ppo.py             # clipped PPO
├── utils/
│   ├── buffers.py         # RolloutBuffer, GAE computation
│   ├── normalizers.py     # RunningMeanStd for obs / returns
│   ├── training.py        # project-owned generic train/eval loops
│   ├── seeding.py         # deterministic SeedSequence spawning
│   └── plotting.py        # learning-curve + track-trajectory plots
└── configs/               # dataclass or YAML experiment configs (net size sweep, etc.)
experiments/               # thin runner scripts / notebooks that use modules under src/
tests/                     # env + geometry unit tests
tracks/                    # circuit definitions (e.g. generated_000.json / oval.json)
```

Design principles to keep it clean and reproducible:

* **Env owns physics + reward; agents own learning.** The env should be
  swappable between Frenet and LiDAR observation without touching agent code —
  do it via a Gymnasium `ObservationWrapper` or an `observation="frenet"|"lidar"`
  constructor flag, so both share one dynamics core.
* **One shared MLP builder** parameterized by `hidden_sizes` is what makes the
  network-size sweep a one-line config change instead of copy-paste.
* **Use a project-owned agent contract.** Define the rollout and update
  interfaces for continuous, vector-valued actions before implementing the first
  agent, and reuse that contract across the project algorithms. Do not inherit a
  lab interface implicitly.
* **Track as data, not code.** Represent circuits as arc-length-sampled
  centerlines + widths in `tracks/*.json`; build a KD-tree (or nearest-segment
  search) once for the point→Frenet projection. This makes the multi-circuit
  variant "load a different file," and lets you procedurally generate tracks for
  domain randomization.
* **Config objects + explicit seeds** for every run, use a SeedSequence.
  Seed env and policy independently.
  Log hyperparameters with results so experiments are reconstructible.
* **Rendering** with `pygame` as an `rgb_array` render mode,
  matching the course's env convention — invaluable for debugging reward hacking
  by *watching* the car.

## Environment / dependencies
- Use the `.venv` env in this repository.
- You may install packages into `.venv` as they become necessary for an approved task; no separate permission is needed for ordinary project dependencies.
- Build the dependency manifest incrementally. Create `requirements.txt` when the first project code is added, and update it in the same task whenever an imported third-party package becomes a project dependency. Use explicit version constraints; create an exact environment freeze before running final experiments.
- Use `gymnasium` to handle the custom environment.
- Use `pygame` for rendering the environment.
- Use `pytorch` to handle the machine learning stuff.


## Commit message convention
`type: summary [ai]`
Examples:
- `feature: add prioritized replay buffer [ai]`
- `fix: correct off-by-one in advantage bootstrap [ai]`
Every commit missing the `[ai]` tag is hand-written by me.
