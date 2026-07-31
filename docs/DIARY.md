# DIARY

## 2026-07-28 — Init
**Task**: Finalization of the project's fundamentals.

**Result**: The `AGENTS.md`, `MDP.md`, `TRACK.md` and `DIARY.md` files.

**Commit**: aab80ff

## 2026-07-28 — Specification audit follow-up
**Task**: Resolve the project-readiness findings selected after a repository-wide audit.

**Result**: Reconciled the reward arithmetic and long-horizon discount; clarified the Markov-like Frenet observation and dynamic curvature preview; documented version 0 physics limitations; specified deterministic track generation, storage, projection, LiDAR, signed progress and lap completion; prohibited implicit reuse of lab code; clarified incremental dependency management; and expanded repository ignore rules.

**Provisional choices**: $\gamma=0.9995$, $w=12m$, $\Delta s_{\text{gen}}=0.5m$, 16 inclusive LiDAR rays over $200°$, and $r_{\max}=100m$. These values are now explicit and may be revised before implementation.

**Files**: `.gitignore`, `AGENTS.md`, `README.md`, `docs/MDP.md`, `docs/TRACK.md`, `docs/DIARY.md`.

**Commit**: 6c239f1

## 2026-07-29 — Environment MVP execution roadmap

**Task**: Expand the initial `PLAN.md` into a practical sequence for building and validating the first racing-environment version.

**Result**: Defined the Phase-1 scope, acceptance criteria, execution rules and 12 sequential implementation milestones. Clarified that manual A/D input controls steering angle, added reward and Gymnasium lifecycle work to the environment scope, deferred grip-limited dynamics and LiDAR, and aligned the suggested flat source layout under `src/`.

**Files**: `PLAN.md`, `AGENTS.md`, `README.md`, `docs/DIARY.md`.

**Commit**: 909a83c

## 2026-07-29 — Flat source layout correction

**Task**: Remove the unwanted intermediate package directory from the proposed source structure.

**Result**: Restored `src/envs`, `src/models`, `src/agents`, `src/utils` and `src/configs` as the direct source directories, and updated every roadmap path and descriptive reference accordingly.

**Files**: `PLAN.md`, `AGENTS.md`, `docs/DIARY.md`.

**Commit**: bedfa4b

## 2026-07-29 — Step 0: Python project scaffold

**Task**: Establish the flat-source Python package structure, dependency manifest, packaging configuration and initial test command.

**Result**: Added importable `src/envs` and `src/configs` packages, configured setuptools and pytest in `pyproject.toml`, pinned the approved Phase-1 dependencies in `requirements.txt`, installed the project in editable mode, and added import smoke tests.

**Dependencies**: Gymnasium 1.3.0, NumPy 2.4.4, Pygame 2.6.1, pytest 9.1.1 and Setuptools 80.10.2. Setuptools remains below version 81 because Pygame 2.6.1 still imports the deprecated `pkg_resources` API.

**Validation**: `pip check` reported no broken requirements; all 5 pytest cases passed; and `envs` and `configs` imported successfully from outside the repository.

**Files**: `PLAN.md`, `pyproject.toml`, `requirements.txt`, `src/envs/__init__.py`, `src/configs/__init__.py`, `tests/test_package.py`, `docs/DIARY.md`.

**Commit**: 4d89e3f

## 2026-07-29 — Step 1: Environment configuration

**Task**: Centralize documented environment constants in typed, immutable and validated configuration objects.

**Result**: Added simulation, vehicle, track-generation, reward and Frenet-observation configuration dataclasses; exported them from `configs`; enforced timing relationships and numeric constraints; and provided deterministic plain-dictionary serialization for future run logging. Training and deferred LiDAR settings were intentionally excluded.

**Validation**: `pip check` reported no broken requirements; all 34 pytest cases passed; and the installed package exposed the configuration successfully outside the repository.

**Files**: `PLAN.md`, `src/configs/environment.py`, `src/configs/__init__.py`, `tests/configs/test_environment_config.py`, `docs/DIARY.md`.

**Commit**: 2d4d41f

## 2026-07-29 — Step 2: Track data and persistence

**Task**: Implement the sampled-track data model and versioned JSON persistence boundary.

**Result**: Added immutable track units and generation metadata, owned read-only float64 sample arrays, strict schema and numeric validation, uniform arc-length and implicit-closing-segment checks, deterministic UTF-8 serialization, and public track loading/saving APIs. Added a circular fixture and malformed fixtures covering schema, ordering, closure and JSON failures. Advanced geometric validation remains isolated to the next roadmap task.

**Validation**: `pip check` reported no broken requirements; all 57 pytest cases passed; byte-stable save/load was verified; and the installed package loaded the fixture successfully outside the repository.

**Files**: `PLAN.md`, `src/envs/track.py`, `src/envs/__init__.py`, `tests/envs/test_track_io.py`, `tests/fixtures/tracks/*.json`, `docs/DIARY.md`.

**Commit**: 2af327a

## 2026-07-29 — Step 3: Track geometry and validation

**Task**: Turn sampled track data into periodic geometric queries and enforce the documented geometric constraints.

**Result**: Added float64 periodic interpolation for centerline position, heading, normal and curvature; sampled left/right boundaries; exact global closest-segment projection backed by a midpoint KD-tree; and validation for length, vehicle curvature limits, centerline and boundary intersections, seam continuity, and nonlocal centerline separation. Track loading and saving now perform geometric validation by default, with an explicit bypass for diagnostic handling of invalid data. Updated the circular fixture to satisfy the project-scale length, width and steering constraints.

**Dependency**: Added SciPy 1.18.0 for periodic cubic splines and the spatial index.

**Validation**: `pip check` reported no broken requirements; all 72 pytest cases passed; source and tests compiled successfully; and diff whitespace validation passed.

**Files**: `PLAN.md`, `requirements.txt`, `src/envs/geometry.py`, `src/envs/track.py`, `src/envs/__init__.py`, `tests/envs/test_track_geometry.py`, `tests/envs/test_track_io.py`, `tests/fixtures/tracks/valid_circle.json`, `docs/DIARY.md`.

**Commit**: 903e96c

## 2026-07-29 — Step 4: Deterministic track generation

**Task**: Generate valid, smoothly sampled tracks deterministically from an explicit seed and configuration.

**Result**: Added project-owned polar checkpoint sampling, periodic cubic-spline smoothing, numerical arc-length resampling, heading and curvature construction, deterministic `SeedSequence` retries, aggregated exhaustion diagnostics, and a thin command-line track generator. Clarified the smoothing method in `TRACK.md` because the earlier bounded-heading-walk description omitted the constants required to implement it without silent hyperparameters.

**Validation**: Fixed seeds generated geometrically valid tracks; identical seeds produced identical serialized data; different seeds produced different geometry; the global NumPy RNG remained untouched; retry exhaustion identified the failed constraint; and generated files loaded through the public API.

**Files**: `PLAN.md`, `docs/TRACK.md`, `src/envs/track_generation.py`, `src/envs/geometry.py`, `src/envs/__init__.py`, `experiments/generate_track.py`, `tests/envs/test_track_generation.py`, `tracks/.gitkeep`, `docs/DIARY.md`.

**Commit**: 96e77e7

## 2026-07-29 — Step 5: Frenet observation geometry

**Task**: Implement Cartesian-to-Frenet projection and the complete Frenet observation geometry.

**Result**: Added signed centerline projection, exact global and temporally coherent local segment searches, physically derived global fallback thresholds, Cartesian reconstruction, wrapped heading error, periodic curvature integration, velocity-dependent curvature preview, physical-unit observation assembly, and unclamped signed periodic progress.

**Validation**: Round trips passed on straights, dense curves and the periodic seam; lateral and heading signs matched the specification; backward progress remained negative; implausible local projections fell back globally; and constant-curvature previews matched analytic values at all supported speeds.

**Files**: `PLAN.md`, `docs/TRACK.md`, `src/envs/observations.py`, `src/envs/geometry.py`, `src/envs/__init__.py`, `tests/envs/test_frenet.py`, `docs/DIARY.md`.

**Commit**: fb38d4f

## 2026-07-31 — Post-refactor coherence audit

**Task**: Reconcile the track, geometry, generation and Frenet-observation code
after simplifying validation and adopting the updated coding conventions.

**Result**: Repaired the public `envs` exports and package imports; completed
the unit-free API naming; aligned the generator CLI and JSON generation
metadata; applied the required class and function documentation style; and
configured Black for the project's Python 3.12 compatibility target. Reworked
the tests around meaningful configuration, persistence, geometry, deterministic
generation and Frenet-observation behaviour, removing obsolete defensive checks
and their unused malformed fixtures.

**Dependency**: Added Black 26.5.1 as the explicitly pinned project formatter.

**Validation**: All 49 tests passed. Black, Ruff, Pyright, `pip check`, source
compilation, editable installation, out-of-tree package imports and an
out-of-tree generator/load smoke test passed.

**Files**: `pyproject.toml`, `requirements.txt`, `src/configs/`,
`src/envs/`, `experiments/generate_track.py`, `tests/`, `docs/TRACK.md`,
`docs/DIARY.md`.

**Commit**: `fix: align track stack after API simplification [ai]`

## 2026-07-31 — Pytest entry-point path fix

**Task**: Make the documented test suite work through both the `pytest`
entry point and `python -m pytest`.

**Result**: Added the repository root explicitly to pytest's import path so
tests can import the project-owned `experiments` namespace independently of
how pytest is launched.

**Validation**: All 49 tests passed through both invocation forms.

**Files**: `pyproject.toml`, `docs/DIARY.md`.

**Commit**: `fix: make pytest invocation independent of entry point [ai]`

## 2026-07-31 — Step 6: Pure kinematic transition

**Task**: Implement the deterministic bicycle-model transition independently
of Gymnasium, lifecycle handling and rendering.

**Result**: Added immutable vehicle-state, normalized-action and physical-control
units; mapped normalized throttle/brake and steering to the documented limits;
and implemented four explicit-Euler physics substeps with per-substep speed
clamping. Each transition returns every intermediate state for the collision and
finish checks introduced by the next lifecycle step.

**Validation**: Covered stationary motion, explicit-Euler straight acceleration,
braking, steering direction, control mapping, speed bounds and four-substep
results.

**Files**: `src/envs/dynamics.py`, `src/envs/__init__.py`,
`tests/envs/test_dynamics.py`, `docs/DIARY.md`.

**Commit**: `feature: add kinematic vehicle transition [ai]`

## 2026-07-31 — Step 7: Episode lifecycle rules

**Task**: Implement collision, finish, progress, reward and time-limit logic
separately from the future Gymnasium environment.

**Result**: Added a lifecycle component that projects every physics substep,
detects point-car crashes, tracks signed unwrapped episode progress, validates
forward finish-gate crossings after a near-full lap, gives crashes precedence,
selects the documented reward branches and distinguishes termination from time
limit truncation. The roadmap now records steps 6 and 7 as completed.

**Validation**: Covered substep collisions, reset/insufficient/backward finish
crossings, valid lap completion, collision precedence, truncation, the stationary
timeout total and immediate crash penalty. The full suite, formatting, linting,
type checking and dependency checks passed.

**Files**: `PLAN.md`, `src/envs/lifecycle.py`, `src/envs/__init__.py`,
`tests/envs/test_episode_lifecycle.py`, `tests/envs/test_reward.py`,
`docs/DIARY.md`.

**Commit**: `feature: add episode lifecycle rules [ai]`
