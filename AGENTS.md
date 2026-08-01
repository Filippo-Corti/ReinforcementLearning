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
2. **One concern per task/commit.** Don't mix algorithm changes with logging/refactor changes in the same diff. Once you have finished your task, always commit to branch `main`.
3. **Document your changes.** Every functionality that is added/modified/removed in the project should be properly documented — not only inside the code and via the git commit, but also in the `docs/DIARY.md` diary file (see that file for examples of the expected format). Reference the commit from rule 2 in the diary entry.
4. **Cite, don't recall.** For anything algorithm-specific, work from the equations/spec provided in this repo or pasted into the prompt rather than from memory of "how X usually works" — implementations vary across papers/codebases.
5. **Do not reuse course-lab code by default.** Code on the `lectures` branch, in lab notebooks, or in earlier exercises is reference material only. Implement project-owned agents and utilities from scratch unless the user explicitly asks to migrate or reuse a specific piece.
6. **No silent hyperparameter choices.** If a value isn't specified anywhere, ask or flag it as a placeholder — don't pick a "reasonable default" silently in core files.
7. **Determinism.** All code involving RNG must accept a `--seed` argument and be reproducible given the same seed + config.

## Coding rules and conventions

1. **Write agile code.** Don't over-engineer validations, immutability, or assertions. Keep the code elegant at the cost of allowing slight misuse. This is a university project, not a production-ready package — test only what is necessary and interesting to test.
2. **Do not refer to plan or step labels in code.** Avoid referring to `plan-0`, `version-0`, `step-0`, or anything of this kind in code, comments, or identifiers. The code must be understandable without knowing the order in which it was implemented.
3. **Do not add units of measure in variable names.**
4. **Class comments.** First give a clear description of what the class does in 1–3 sentences, then list the Fields:

   ```python
   class ExampleClass:
     """
     This class represents an example of a how a comment should be written 
     for a class in this repository.

     Fields:
       * field_name: The description of the field
       ...
     """
   ```

5. **Method/function comments.** Always write the docstring body on its own line(s), not on the same line as the opening `"""`:

   ```python
   def func(params):
     """
     Do this
     """
   ```

   not:

   ```python
   def func(params):
     """Don't do this"""
   ```

6. **Use Black formatting.** Always.
7. **Organize modules by responsibility.** Split a module when it owns several
   independently named concepts, and place each concept in the package that owns
   its domain. Do not split merely to reduce line count; every file and package
   should have a clear, stable purpose.
8. **Use names that reveal role and timescale.** Distinguish data from the
    processor that creates it, physical transitions from lifecycle outcomes, and
    per-action results from complete-episode summaries. Avoid generic names such
    as `Result`, `Manager`, or `Geometry` when a more specific role is known.
9. **Preserve semantic data until framework boundaries.** Represent meaningful
    internal records with named dataclasses rather than anonymous arrays or
    tuples. Convert to NumPy arrays only at numerical or framework interfaces
    such as Gymnasium observations.
10. **Centralize shared primitives.** Search the repository before adding small
    geometry, conversion or typing helpers. Put truly shared primitives in a
    neutral package and remove local copies in the same change.
11. **Lead APIs with their primary operation.** Put the main public method first,
    keep secondary helpers lower in the class, and make implementation-only
    operations private. Public names and docstrings should make return values and
    mutation scope clear without requiring the caller to read the implementation.
12. **Explain framework-required declarations.** Add a short explanation for
    otherwise opaque class attributes, metadata or hooks.

### Suggested code structure:
This is the suggested layout. Feel free to adapt it as you wish, but make sure to update this if you ever choose to:

```
src/
├── envs/
│   ├── geometry/
│   │   ├── angles.py      # reusable angular operations
│   │   ├── projection.py  # closest-segment spatial queries
│   │   └── segments.py    # reusable finite-segment operations
│   ├── observations/
│   │   └── frenet.py      # Frenet observation data and observer
│   ├── racing/
│   │   ├── environment.py # RacingEnv(gym.Env): Gymnasium reset/step shell
│   │   ├── lifecycle.py   # progress, reward, termination and truncation
│   │   └── rendering.py   # optional Pygame human/rgb_array presentation
│   ├── tracks/
│   │   ├── errors.py      # track-specific validation and generation errors
│   │   ├── generation.py  # deterministic procedural generation
│   │   ├── track.py       # sampled tracks, derived geometry and persistence
│   │   └── validation.py  # track-specific geometry constraints
│   ├── vehicle/
│   │   ├── controls.py    # normalized actions and physical controls
│   │   ├── kernel.py      # kinematic transition kernel
│   │   └── state.py       # vehicle state
│   └── types.py           # numerical types shared across environment domains
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

### Environment / dependencies
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
