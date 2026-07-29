# Project Plan — Phase 1: Racing Environment MVP

## Goal

Build a reproducible, testable version-0 racing environment with:

- deterministic procedural track generation;
- persistent track save/load support;
- kinematic car dynamics;
- Frenet observations;
- reward, termination and truncation logic;
- Pygame rendering; and
- a manual driving script for end-to-end validation.

This phase establishes the environment on which the reinforcement-learning
algorithms will later operate. It does **not** implement an RL agent.

## Clarifications to the Initial Plan

The following points resolve ambiguities in the original draft:

1. “Fully functioning physics simulation” means the version-0 kinematic model
   specified in [`docs/MDP.md`](docs/MDP.md), not a complete vehicle-dynamics
   simulator.
2. W/S control normalized throttle and braking. A/D control normalized steering
   angle, not angular acceleration.
3. The environment must produce rewards and follow Gymnasium's
   `terminated`/`truncated` contract; advancing the physical state alone is not a
   complete environment step.
4. Version 0 intentionally excludes lateral-grip limits, aerodynamic drag, tire
   slip, steering-rate limits and a finite collision footprint. In particular,
   “centripetal force” is not added in this phase. A later grip-limited version
   will constrain feasible lateral acceleration.
5. Only the Frenet observation is implemented in this phase. LiDAR remains a
   later variant.
6. The car is a point for collision detection, even if the renderer draws a
   larger marker so its heading is visible.
7. Version 0 resets at the canonical start line with zero speed. Randomized
   starts are a later curriculum feature.
8. Training policies, neural networks, rollout buffers and experiment sweeps are
   outside this phase.

## Authoritative Specifications

Implementation decisions must agree with:

- [`docs/MDP.md`](docs/MDP.md) for state, action, dynamics, reward and episode
  lifecycle;
- [`docs/TRACK.md`](docs/TRACK.md) for track geometry, persistence, Frenet
  projection, progress and lap completion;
- [`AGENTS.md`](AGENTS.md) for workflow, testing, determinism and dependency
  rules.

If implementation exposes a missing or contradictory requirement, update the
relevant specification and obtain confirmation before continuing. Do not hide a
new constant in source code.

## Definition of Done

Phase 1 is complete when all of the following are true:

- A seed and explicit configuration generate a valid circuit deterministically.
- A generated circuit survives a JSON save/load round trip without changing its
  runtime geometry.
- The implementation rejects malformed or geometrically invalid track files.
- Cartesian points can be converted to Frenet coordinates accurately, including
  near the periodic seam.
- The car follows the four-substep kinematic transition defined in `MDP.md`.
- The environment returns the documented Frenet observation and reward.
- Finish, crash and time-limit outcomes produce the correct Gymnasium flags.
- Repeating the same seeded action sequence produces identical trajectories and
  rewards.
- `gymnasium.utils.env_checker.check_env` passes.
- `human` and `rgb_array` rendering modes work.
- A user can launch a script and drive with W/S/A/D, reset, and quit cleanly.
- Unit and integration tests pass from a fresh project environment.
- The README and diary explain how to install, run and validate the environment.

## Execution Rules

Each numbered step below is a separate task and commit. Before implementing a
step:

1. restate the scope and list the files to be touched;
2. resolve any decision explicitly marked as pending;
3. obtain confirmation;
4. implement the smallest complete increment;
5. run the step's validation gate;
6. update `docs/DIARY.md`; and
7. commit to `main` using the project commit convention.

Do not start a later step while an earlier validation gate is failing.

## Practical Execution Steps

### 0. Create the Python Project Scaffold

**Status:** Completed on 2026-07-29.

**Objective:** Establish importable project structure and a reproducible test
command before adding behaviour.

**Work:**

- Use the flat source layout rooted at `src/`.
- Create the initial `envs` and `configs` packages directly under `src/`.
- Create `requirements.txt` with explicit constraints for dependencies used by
  this phase.
- Add packaging and pytest configuration.
- Install required packages into `.venv`.
- Add a minimal import test.

**Initial dependencies:** NumPy, Gymnasium, Pygame and pytest. Add SciPy only if
the confirmed geometry implementation uses its spline or spatial-index tools.
PyTorch is not required until policy implementation.

**Expected files:**

- `pyproject.toml`
- `requirements.txt`
- `src/envs/__init__.py`
- `src/configs/__init__.py`
- `tests/test_package.py`
- `docs/DIARY.md`

**Validation gate:**

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest
```

### 1. Centralize the Environment Configuration

**Status:** Completed on 2026-07-29.

**Objective:** Give every documented physical and track constant one typed,
validated source of truth.

**Work:**

- Define immutable configuration objects for simulation, vehicle, track
  generation, reward and observation settings.
- Populate them only with values already approved in `MDP.md` and `TRACK.md`.
- Validate ranges and relationships such as
  `agent_dt / physics_dt == physics_substeps`.
- Support serialization to a plain dictionary for logging.
- Do not add training hyperparameters.

**Expected files:**

- `src/configs/environment.py`
- `tests/configs/test_environment_config.py`
- `docs/DIARY.md`

**Validation gate:**

- Defaults match the documentation exactly.
- Invalid timestep ratios, widths, ranges and limits raise clear errors.
- Configuration serialization is deterministic.

### 2. Implement the Track Data Model and JSON Persistence

**Status:** Completed on 2026-07-29.

**Objective:** Load, validate and save already-sampled tracks before attempting
procedural generation.

**Work:**

- Define a track object containing metadata, width, sample spacing, total length
  and sampled `s/x/y/heading/curvature` arrays.
- Implement the versioned JSON schema from `TRACK.md`.
- Validate schema version, units, finite numeric values, shapes, increasing
  arc-length samples and closing-segment consistency.
- Implement deterministic serialization.
- Add a small hand-authored circular or oval fixture for tests only.

**Expected files:**

- `src/envs/track.py`
- `tests/envs/test_track_io.py`
- `tests/fixtures/tracks/valid_circle.json`
- `tests/fixtures/tracks/invalid_*.json`
- `docs/DIARY.md`

**Validation gate:**

- Valid fixture loads successfully.
- Save/load preserves all runtime fields.
- Unknown schema versions and invalid files are rejected.
- Serialization output is stable.

### 3. Implement Track Geometry and Validation

**Status:** Completed on 2026-07-29.

**Objective:** Turn sampled track data into reliable geometric queries.

**Work:**

- Interpolate centerline position, heading, normal and local curvature
  periodically in `s`.
- Construct left and right boundary segments.
- Validate closure, tangent/curvature seam continuity, length, curvature,
  self-intersections, boundary intersections and minimum nonlocal separation.
- Build the global segment-search structure used by projection.
- Keep geometry calculations in `float64`; convert observations later at the
  environment boundary.

**Expected files:**

- `src/envs/track.py`
- optional `src/envs/geometry.py` if the helpers become substantial
- `tests/envs/test_track_geometry.py`
- `docs/DIARY.md`

**Validation gate:**

- Analytic circle tests confirm position, tangent, normal and curvature.
- Queries are continuous across the periodic seam.
- Known self-intersecting and over-curved tracks fail validation.
- Boundary construction preserves the documented left/right convention.

### 4. Implement Deterministic Procedural Track Generation

**Status:** Completed on 2026-07-29.

**Objective:** Generate valid random tracks from a seed and save them as data.

**Work:**

- Implement the checkpoint-and-smoothing generator defined in `TRACK.md`.
- Use `numpy.random.SeedSequence` and a local generator; never use global random
  state.
- Resample the smoothed closed curve at the configured arc-length spacing.
- Compute heading and curvature from the final curve.
- Apply every documented validity check.
- Retry deterministically up to `max_attempts`, then raise a diagnostic error.
- Provide a thin command-line generator that writes `tracks/*.json`.

**Expected files:**

- `src/envs/track_generation.py`
- `experiments/generate_track.py`
- `tests/envs/test_track_generation.py`
- `tracks/.gitkeep`
- `docs/DIARY.md`

**Validation gate:**

- Same seed and configuration produce identical serialized data.
- A sample of fixed test seeds produces valid tracks.
- Different seeds produce different geometry.
- Retry exhaustion reports which constraints failed.
- Generated tracks load through the public loader.

### 5. Implement Cartesian-to-Frenet Projection and Preview

**Status:** Completed on 2026-07-29.

**Objective:** Produce the complete version-0 observation geometry.

**Work:**

- Project a Cartesian point onto the closest centerline segment.
- Implement signed lateral distance with positive values on the left.
- Implement global search plus the temporally coherent local-window
  optimization and safe fallback.
- Wrap heading error to `[-pi, pi)`.
- Compute the velocity-dependent average-curvature preview by integrating the
  preprocessed curvature table.
- Implement signed periodic progress without monotonic clamping.

**Expected files:**

- `src/envs/track.py`
- `src/envs/observations.py`
- `tests/envs/test_frenet.py`
- `docs/DIARY.md`

**Validation gate:**

- `frenet(xy_from_frenet(s, d))` round-trips on straights, curves and the seam.
- Lateral-distance and heading-error signs match the specification.
- Backward motion produces negative progress.
- Implausible local projections trigger global fallback.
- Dynamic preview agrees with analytic constant-curvature cases.

### 6. Implement the Pure Kinematic Transition

**Objective:** Advance vehicle state independently of Gymnasium, reward and
rendering.

**Work:**

- Define the vehicle state and normalized two-dimensional control action.
- Map throttle/brake and steering inputs to physical controls.
- Apply four explicit-Euler physics substeps per agent action.
- Enforce speed bounds after each substep.
- Expose intermediate substep poses so collision and finish checks can run at
  physics frequency.
- Keep the function deterministic and free of global mutable state.

**Expected files:**

- `src/envs/dynamics.py`
- `tests/envs/test_dynamics.py`
- `docs/DIARY.md`

**Validation gate:**

- Zero-speed/zero-action state remains stationary.
- Straight acceleration and braking match analytic expectations.
- Positive and negative steering turn in the documented directions.
- Speed never leaves `[0, v_max]`.
- One action is exactly four physics substeps.

### 7. Implement Collision, Finish, Progress and Reward

**Objective:** Implement episode lifecycle rules separately from the Gymnasium
shell.

**Work:**

- Detect point-car collision after every physics substep.
- Detect a valid forward finish-gate crossing with the required accumulated
  progress.
- Give collision precedence when collision and finish coincide.
- Compute signed progress and the documented reward branches.
- Distinguish true termination from the 5000-step time-limit truncation.
- Return diagnostic values needed in the future `info` dictionary.

**Expected files:**

- `src/envs/racing.py` or a focused lifecycle helper module
- `tests/envs/test_episode_lifecycle.py`
- `tests/envs/test_reward.py`
- `docs/DIARY.md`

**Validation gate:**

- Stationary timeout, immediate crash and nominal lap totals match `MDP.md`.
- Crossing the line immediately after reset does not finish.
- Backward and insufficient-progress crossings do not finish.
- A valid forward full-lap crossing terminates.
- Time limit truncates without marking the MDP state terminal.

### 8. Assemble and Register `RacingEnv`

**Objective:** Expose the tested components through a compliant Gymnasium
environment.

**Work:**

- Implement `RacingEnv` with `reset`, `step`, `close` and seeded track/reset
  handling.
- Define action and Frenet observation spaces with matching shapes and dtypes.
- Return useful `info` fields such as wrapped/unwrapped progress, collision,
  lap completion, elapsed simulated time and track seed.
- Support loading a saved track and generating from a supplied seed/config.
- Register a versioned environment ID only if registration improves the public
  usage; direct construction remains supported.

**Expected files:**

- `src/envs/racing.py`
- `src/envs/__init__.py`
- `tests/envs/test_racing_env.py`
- `docs/DIARY.md`

**Validation gate:**

- Gymnasium `check_env` passes.
- Observation values belong to the declared space, including terminal
  observations.
- Same seed and action sequence reproduce observations, rewards and flags.
- Reset clears all episode caches and counters.

### 9. Add Pygame Rendering

**Objective:** Visualize the same state and geometry used by the environment.

**Work:**

- Support `render_mode="human"` and `render_mode="rgb_array"`.
- Draw track boundaries, centerline, finish gate and a heading-visible car
  marker.
- Keep rendering optional so tests and training run headlessly.
- Ensure `close()` releases Pygame resources and repeated environment creation
  is safe.
- Keep camera and colour choices out of physics calculations.

**Expected files:**

- `src/envs/rendering.py`
- `src/envs/racing.py`
- `tests/envs/test_rendering.py`
- `docs/DIARY.md`

**Validation gate:**

- `rgb_array` returns an image with the declared shape and dtype.
- Headless rendering smoke test passes.
- Human rendering opens, updates and closes without leaking resources.

### 10. Add the Manual Driving Script

**Objective:** Provide the requested end-to-end human validation tool.

**Controls:**

- W: positive throttle.
- S: braking.
- A: steer left.
- D: steer right.
- R: reset with the current track.
- Esc or window close: exit cleanly.

Simultaneous throttle/brake-axis and steering-axis input is allowed. Releasing a
key returns that control axis to zero.

**Work:**

- Add CLI arguments for track path or generation seed.
- Run at the documented 25 agent actions per simulated second.
- Display basic diagnostics such as speed, progress, reward and terminal reason.
- Reset cleanly after finish/crash on user request; do not hide terminal states
  through an automatic reset.

**Expected files:**

- `experiments/manual_drive.py`
- `tests/experiments/test_manual_drive.py` for argument parsing and import safety
- `README.md`
- `docs/DIARY.md`

**Validation gate:**

```powershell
.\.venv\Scripts\python.exe experiments\manual_drive.py --seed 0
```

- Controls match the documented action signs.
- The program remains responsive and exits cleanly.
- A user can observe collision, truncation and lap-completion behaviour.

### 11. Run the Phase-1 Acceptance Pass

**Objective:** Prove that the environment MVP is ready for agent development.

**Work:**

- Run the entire automated suite from a fresh environment.
- Generate, save, reload and render at least one fixed-seed track.
- Run a deterministic action-sequence replay and compare complete trajectories.
- Execute Gymnasium's environment checker.
- Perform the manual driving smoke test.
- Record test commands, dependency versions and known version-0 limitations.
- Update the README quickstart and diary.

**Expected files:**

- `README.md`
- `docs/DIARY.md`
- test or fixture corrections only if acceptance exposes a defect

**Validation gate:**

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest
```

All automated checks pass, the manual script works, and no undocumented
hyperparameter or physics choice remains in the phase-1 implementation.

## Deferred Work

After Phase 1, create new plans rather than silently extending this one:

1. lateral-grip constraint and finite vehicle footprint;
2. reward validation/tuning with the revised dynamics;
3. project-owned REINFORCE baseline;
4. A2C/GAE and PPO;
5. network-size experiments and formal convergence protocol;
6. LiDAR observation;
7. randomized starts and curricula;
8. multi-track training and held-out-track generalization.
