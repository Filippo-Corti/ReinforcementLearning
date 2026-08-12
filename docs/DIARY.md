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

## 2026-07-31 — Step 8: RacingEnv assembly

**Task**: Assemble the track, kinematic transition and lifecycle rules into a
Gymnasium-compatible racing environment.

**Result**: Added `RacingEnv` with deterministic reset-seed track generation,
saved-track loading, canonical zero-speed resets, continuous float32 action and
Frenet observation spaces, terminal-state handling, and diagnostic progress,
outcome, elapsed-time and track-seed information. The environment remains
render-free until the dedicated rendering step. Also corrected a KD-tree
floating-point boundary case exposed when resetting from a saved track.

**Validation**: Gymnasium's `check_env` passed; seeded action sequences matched;
saved tracks loaded; reset cleared lifecycle state; terminal observations stayed
within the declared space; and the complete suite, formatting, linting, type
checking and dependency checks passed. Gymnasium reports only advisory warnings
for the intentionally unbounded Frenet dimensions.

**Files**: `PLAN.md`, `src/envs/racing.py`, `src/envs/lifecycle.py`,
`src/envs/geometry.py`, `src/envs/__init__.py`,
`tests/envs/test_racing_env.py`, `docs/DIARY.md`.

**Commit**: `feature: assemble Gymnasium racing environment [ai]`

## 2026-07-31 — Step 9: Pygame rendering

**Task**: Visualize the environment state with optional human and RGB-array
Pygame rendering.

**Result**: Added a display-only renderer that fits the track boundaries to an
800×800 camera, draws the road, boundaries, centerline, canonical finish gate
and heading-visible car marker, and exposes `human` and `rgb_array` modes from
`RacingEnv`. Renderer creation is lazy, seeded reset safely discards a stale
renderer, and repeated close calls release the display without affecting physics.
Documented the canvas and camera choices in `TRACK.md` so they are explicit.

**Validation**: RGB frames have the documented shape and dtype and change after
state transitions; human rendering opened and closed under SDL's dummy driver;
Gymnasium's render checker passed; and the complete suite, formatting, linting,
type checking and dependency checks passed.

**Files**: `PLAN.md`, `docs/TRACK.md`, `src/envs/rendering.py`,
`src/envs/racing.py`, `src/envs/__init__.py`,
`tests/envs/test_rendering.py`, `docs/DIARY.md`.

**Commit**: `feature: add Pygame racing renderer [ai]`

## 2026-07-31 — Environment package refactor

**Task**: Reorganize the growing environment implementation into smaller,
domain-focused packages without changing its public behaviour.

**Result**: Grouped the Gymnasium shell, lifecycle and renderer under `racing`;
track persistence, generation, interpolation, projection, validation and Frenet
observations under `tracks`; and vehicle dynamics under `vehicle`. Split the
former geometry module into focused geometry, projection and validation modules,
kept the top-level `envs` exports stable, updated the generator import and revised
the documented source layout.

**Validation**: All 74 tests passed. Black, Ruff, Pyright, dependency checks,
source compilation and diff whitespace validation passed.

**Files**: `AGENTS.md`, `src/envs/`, `experiments/generate_track.py`,
`docs/DIARY.md`.

**Commit**: `refactor: organize environment packages [ai]`

## 2026-07-31 — Interactive track viewer

**Task**: Add a small experiment for inspecting generated and saved tracks.

**Result**: Added an import-safe CLI that opens the existing human renderer at
the canonical start pose, accepts exactly one of a deterministic seed or saved
track path, refreshes at the environment decision rate and exits on Escape or a
window-close event. Added README commands for both track sources.

**Validation**: Covered CLI source selection, environment construction and main
delegation without entering the blocking display loop. The full automated and
static validation suite passed.

**Files**: `experiments/view_track.py`,
`tests/experiments/test_view_track.py`, `README.md`, `docs/DIARY.md`.

**Commit**: `feature: add interactive track viewer [ai]`

## 2026-07-31 — Main-branch continuous integration

**Task**: Run the repository's automated validation on GitHub for every push to
`main`.

**Result**: Added a least-privilege GitHub Actions workflow on Ubuntu and Python
3.13. It installs the pinned project environment with pip caching, then runs
dependency validation, Black, Ruff, Pyright, source compilation, pytest and
whitespace checks. This is continuous integration only; the project currently
has no deployment target.

**Validation**: Verified the workflow structure and reran every configured CI
command locally.

**Files**: `.github/workflows/ci.yml`, `docs/DIARY.md`.

**Commit**: `ci: validate every push to main [ai]`

## 2026-08-01 — Step 10: Manual driving experiment

**Task**: Add the requested interactive, end-to-end environment validation tool.

**Result**: Added an import-safe keyboard driver for deterministic generated or
saved tracks. W/S control the signed throttle/brake axis and A/D the signed
steering axis; R resets the same track; Escape and the window close event exit
cleanly. It advances the environment at the documented 25 decision steps per
second and exposes speed, progress, reward and terminal status in the window
title. Terminal states remain visible until a user reset or exit.

**Validation**: Added parser, environment-construction, action-sign,
terminal-loop and main-delegation tests. The complete automated and static
validation suite passed.

**Files**: `PLAN.md`, `experiments/manual_drive.py`,
`tests/experiments/test_manual_drive.py`, `README.md`, `docs/DIARY.md`.

**Commit**: `feature: add manual driving experiment [ai]`

## 2026-08-01 — Step 11: Phase-1 acceptance pass

**Task**: Verify the complete racing-environment MVP before beginning agent
development.

**Result**: Added the import-safe `experiments/phase1_acceptance.py` runner. It
executes dependency, formatting, linting, type, compilation, pytest and diff
whitespace checks; generates, saves, reloads and RGB-renders an explicitly
seeded track; compares complete outputs from repeated fixed action replays;
runs Gymnasium's environment checker; and starts then exits the manual driver
using SDL's dummy backend. Added focused tests for the acceptance helpers and
documented the command and pinned dependency versions in the README.

**Commands**: `python -m pip check`, `python -m black --check src experiments
tests`, `python -m ruff check src experiments tests`, `python -m pyright src
tests experiments`, `python -m compileall -q src experiments tests`, `python
-m pytest`, `git diff --check`, and `python experiments/phase1_acceptance.py
--seed 0`. The recorded run used Python 3.13.12 from the repository `.venv`;
the README gives the fresh-environment invocation.

**Known version-0 limitations**: The car is a collision point and uses a
kinematic bicycle model without lateral grip, aerodynamic drag, tire slip,
steering-rate limits or a finite vehicle footprint. LiDAR, randomized starts,
learning agents and multi-circuit training remain deferred.

**Files**: `PLAN.md`, `README.md`, `experiments/phase1_acceptance.py`,
`tests/experiments/test_phase1_acceptance.py`, `docs/DIARY.md`.

**Commit**: `test: add phase one acceptance runner [ai]`

## 2026-08-01 — Manual reset display fix

**Task**: Prevent the manual-driving experiment from losing its Pygame display
when the user presses R.

**Result**: The reset path now renders the reset pose immediately and skips the
control step for that frame. This recreates the display closed by environment
reset before Pygame keyboard state is queried, while keeping the reset state
visible for a complete frame.

**Validation**: Added a regression test that resets, resumes keyboard-controlled
stepping and exits. The complete phase acceptance runner passed.

**Files**: `experiments/manual_drive.py`,
`tests/experiments/test_manual_drive.py`, `docs/DIARY.md`.

**Commit**: `fix: restore display after manual reset [ai]`

## 2026-08-01 — Environment responsibility refactor

**Task**: Resolve the linked source TODO review as one architectural change and
record conventions that prevent the same organizational drift.

**Result**: Split vehicle state, controls and the kinematic kernel; moved reusable
angles, segment operations and polyline projection into a shared geometry
package; introduced a dedicated observations package with named
`FrenetObservation` data and a primary `FrenetObserver.observe` API; combined
sampled tracks and derived runtime geometry under `TrackWithGeometry`; scoped
track errors to the tracks package; and renamed action-level, transition and
Pygame-rendering types to reveal their roles. `RacingEnv` now accepts one
prepared `TrackWithGeometry`, with deterministic generation and file loading
performed by callers. Experiments and tests use the same construction boundary.

**Documentation**: Added the runtime track-preparation API to `TRACK.md`, updated
the source-layout reference, and added review-inventory, responsibility,
orchestration, naming, semantic-data, shared-primitive, API-ordering and
framework-explanation rules to `AGENTS.md`.

**Validation**: The Phase-1 acceptance runner passed with seed 0. It included
dependency checking, Black, Ruff, Pyright, compilation, 86 tests, whitespace
validation, deterministic replay, track persistence, RGB rendering, Gymnasium
conformance and the manual-driver SDL smoke test. Gymnasium emitted only its
existing advisory warnings for intentionally unbounded observation dimensions
and direct, unregistered construction.

**Files**: `AGENTS.md`, `docs/TRACK.md`, `src/configs/`, `src/envs/`,
`experiments/`, `tests/`, `docs/DIARY.md`.

**Commit**: `68038bb`

## 2026-08-03 — Phase-2 learned-baseline roadmap

**Task**: Evaluate the completed Phase-1 repository, archive its execution
roadmap and define the next phase without advancing the environment model ahead
of observed learning evidence.

**Result**: Confirmed that every Phase-1 plan step is represented in the clean
`main` worktree and moved the completed roadmap to
`docs/old-plans/phase-1-racing-environment-mvp.md`. Replaced `PLAN.md` with the
Phase-2 learned-baseline and grip-validation roadmap. Its additive gates first
specify and test metrics, seed streams, neural-policy primitives, rollouts and
vanilla REINFORCE; then require a learned version-0 racing result and a recorded
curvature-conditioned speed/throttle diagnosis before lateral-grip work can
begin. The same reward and training comparison must be repeated after the
minimal approved grip change, and a learned value baseline remains a later,
separately measured algorithm increment. A2C, PPO and the network-size study are
explicitly deferred.

**Validation**: The first acceptance attempt exposed local environment drift:
Black was declared in `requirements.txt` but absent from `.venv`. Installed the
already pinned `black==26.5.1` without changing the dependency manifest, then ran
`python experiments/phase1_acceptance.py --seed 0`. Dependency, formatting,
linting, type, compilation and whitespace checks passed, as did all 86 tests,
deterministic replay, track persistence, RGB rendering, Gymnasium conformance
and the manual-driver SDL smoke test. Gymnasium retained only its documented
advisory warnings about unbounded observation dimensions and direct,
unregistered construction.

**Files**: `PLAN.md`,
`docs/old-plans/phase-1-racing-environment-mvp.md`, `docs/DIARY.md`.

**Commit**: `docs: define phase two learning roadmap [ai]`

## 2026-08-09 — Experiment-ready Phase-2 roadmap

**Task**: Revisit Phase 2 after fixing the project's experimental direction and
formally specify the two planned studies.

**Result**: Replaced the learned-baseline-only roadmap with an experiment-ready
implementation phase covering shared neural and rollout primitives,
REINFORCE, A2C+GAE, PPO, reproducible artifacts, analysis, the conditional grip
gate, LiDAR, deterministic multi-track scheduling and reduced end-to-end smoke
matrices. The full measurement runs remain outside Phase 2; acceptance instead
proves that all configurations execute and that their manifests contain no
unresolved scientific choices.

Added `EXPERIMENT.md` as the authoritative empirical protocol. Experiment 1 is
a 3-algorithm by 3-actor-size design with five paired training roots on one
fixed Frenet circuit. Only actor capacity varies across sizes while critic
capacity stays fixed. Experiment 2 selects the smallest adequate PPO actor by a
predeclared Experiment 1 rule, then compares paired Frenet and LiDAR runs on
procedural training circuits and disjoint held-out circuits. The protocol
defines experimental units, pilot/measurement separation, the pre-measurement
grip decision, deterministic evaluation, convergence and censoring, task and
computational measures, optimization diagnostics, seed-level aggregation,
required reports, artifacts and limitations. Updated the README to summarize
the approved sequence and link to the dedicated protocol.

**Validation**: `git diff --check` passed and all 86 tests passed. Gymnasium
retained only its two documented advisory warnings about the intentionally
unbounded Frenet observation dimensions.

**Files**: `PLAN.md`, `EXPERIMENT.md`, `README.md`, `docs/DIARY.md`.

**Commit**: `docs: define experiment-ready phase two [ai]`

## 2026-08-10 — Learning contract and frozen protocol baseline

**Task**: Complete Step 0 of the Phase-2 roadmap by translating the course
policy-gradient notation into an implementation-complete learning specification
and resolving the experiment fields that must not be chosen per run.

**Result**: Added the bounded diagonal-Gaussian policy and stable transformed
log-probability, deterministic action, shared network and numerical conventions,
REINFORCE return-to-go, A2C+GAE targets and PPO clipped losses. The contract
distinguishes true termination, time-limit truncation and rollout cuts, and gives
short project-specific pseudocode for all three agents. Added a dated protocol
registry covering finite learning-rate calibration, interaction budgets,
cadences, seed namespaces, fixed-track selection, convergence, the conditional
grip gate, multi-track splits, deterministic execution and artifact retention.
Corrected experiment links and removed README advice that conflicted with the
canonical start and conditional-grip design.

**Validation**: All scoped Markdown links resolved and `git diff --check`
passed. The 100 deterministic fixed-track candidate identities produced 100
unique generator seeds, with 85 candidates satisfying the registered geometry
thresholds. `python experiments/phase1_acceptance.py --seed 0` passed dependency,
Black, Ruff, Pyright, compilation, whitespace, deterministic replay, rendering,
Gymnasium conformance and all 86 tests. Gymnasium retained only its existing
advisory warnings for unbounded observation limits and direct construction.

**Files**: `docs/LEARNING.md`, `docs/EXPERIMENT.md`, `README.md`, `PLAN.md`,
`docs/DIARY.md`.

**Commit**: `docs: specify learning contract and freeze protocol [ai]`

## 2026-08-10 — Explain and reorganize the learning protocol

**Task**: Resolve the author's inline review notes on the learning and experiment
references without discarding the scientific controls introduced in Step 0.

**Result**: Rewrote `LEARNING.md` to explain why the bounded Gaussian policy is
needed, how its transformed log-probability works, and exactly how the actor and
critic are constructed. Replaced numeric episode masks with piecewise boolean
cases, explained observation normalization and gradient clipping, introduced
each algorithm in relation to the previous one, and expanded all three
pseudocode blocks into complete collection/target/update procedures. Added a
provenance table that distinguishes course equations, MDP timescale choices,
original-paper configurations and project engineering safeguards.

Reorganized `EXPERIMENT.md` so pre-experiment configuration is clearly separate
from scientific evidence, Experiment 1 is presented completely before Experiment
2, and each experiment lists its recorded data and reported outcomes directly.
Explained seed streams with a concrete example, replaced abbreviated run-purpose
labels with descriptive names, documented the rationale for hand-chosen
constants, and changed the supported reported execution device to the available
NVIDIA GeForce RTX 2060 while retaining CPU environment simulation. Aligned the
roadmap, README and MDP discount wording with the revised contract.

**Validation**: All inline TODO notes were removed after resolution. All scoped
Markdown links resolved, display-math delimiters were paired and
`git diff --check` passed. `python experiments/phase1_acceptance.py --seed 0`
passed dependency, Black, Ruff, Pyright, compilation, whitespace, deterministic
replay, rendering, Gymnasium conformance and all 86 tests. Gymnasium retained
only its existing advisory warnings.

**Files**: `docs/LEARNING.md`, `docs/EXPERIMENT.md`, `docs/MDP.md`, `PLAN.md`,
`README.md`, `docs/DIARY.md`.

**Commit**: `docs: clarify learning and experiment contracts [ai]`

## 2026-08-10 — Clarify learning notation and safeguards

**Task**: Remove the remaining ambiguities in the learning reference around
squashed-Gaussian sampling, normalization, gradient clipping and autograd
notation.

**Result**: Defined the Hadamard product componentwise and explained the
reparametrization trick, including why these score-function estimators detach
the sampled latent action. Explained change of variables as probability-mass
preservation under the `tanh` transformation and retained only the mathematical
Jacobian correction, leaving its numerical evaluation to the implementation.
Replaced Welford normalization with componentwise running counts, sums and
squared sums. Distinguished the all-algorithm gradient-norm safeguard from PPO's
ratio clipping and disabled value clipping. Renamed the mathematical
`stop` operator to `detach` to match PyTorch.

**Validation**: All scoped Markdown links resolved, display-math delimiters were
paired, the removed terminology and Jacobian evaluation identity were absent,
and `git diff --check` passed. `python experiments/phase1_acceptance.py --seed 0`
passed dependency, Black, Ruff, Pyright, compilation, whitespace, deterministic
replay, rendering, Gymnasium conformance and all 86 tests. Gymnasium retained
only its existing advisory warnings.

**Files**: `docs/LEARNING.md`, `docs/DIARY.md`.

**Commit**: `docs: clarify learning implementation notation [ai]`

## 2026-08-10 — Training configuration and independent seed streams

**Task**: Implement Phase-2 Step 1 so every approved learning choice and source
of randomness has a machine-readable contract before neural models are added.

**Result**: Added frozen, stable-serializable configurations for the three named
actor sizes, fixed critic, optimizer and normalization conventions, each agent's
collection and update settings, training/evaluation/logging schedules, execution
policy, learning-rate calibration and both experiment matrices. Learning rates
remain finite pre-experiment candidates rather than implicit final defaults.
Added the full namespace registry and seven fixed child seed streams, with fresh
NumPy and PyTorch generators that do not consume siblings or mutate global RNG
state. Added explicit process-wide PyTorch determinism setup and documented its
cross-platform limits. Declared the supported PyTorch 2.11 release series.

**Review**: After delegated implementation, added the missing critic hidden-layer
initialization gain and disabled scheduler setting, exposed local PyTorch
generators and track seeds, and separated the training/checkpoint schedule from
evaluation-owned settings.

**Validation**: Black, Ruff, Pyright, nine focused configuration/seeding tests,
`pip check` and `git diff --check` passed. The Phase-1 acceptance runner passed
all 95 tests; Gymnasium retained only its existing advisory warnings.

**Files**: `src/configs/serialization.py`, `src/configs/training.py`,
`src/configs/experiments.py`, `src/configs/__init__.py`, `src/utils/seeding.py`,
`src/utils/__init__.py`, `requirements.txt`, focused tests, `docs/LEARNING.md`,
`PLAN.md`, `docs/DIARY.md`.

**Commit**: `feature: add training configs and seed streams [ai]`

## 2026-08-10 — Metrics, artifacts and driving references

**Task**: Implement Phase-2 Step 2 so environment behaviour, learning updates,
resource use and run provenance are observable before neural agents are added.

**Result**: Added versioned semantic records for transitions, episodes,
evaluations, optimizer updates, timing and resources, with explicit run category,
metric scope and completion/crash/time-limit outcomes. Added atomic JSON snapshot
writes, durable and validated JSONL streams, category/schema rejection, required
run directories, incomplete-run detection, manifest checksums and reproducibility
metadata covering Git, dependencies, platform, execution and available PyTorch/
CUDA context. Added isolated random-action and exact scripted Frenet reference
policies plus a seeded CLI that generates and saves its circuit, writes complete
artifacts and keeps all reference interactions outside the training counter.

**Review**: After delegated implementation, required lifecycle and next-state
fields on transitions, added structured speed/control episode summaries and
explicit optimizer diagnostics, required every metric row to declare the run
category, prevented corrupt JSONL streams from being completed, separated
evaluation and persistence timing, exposed total parameter count and ignored
routine `results/` trees.

**Validation**: Black, Ruff, Pyright, 13 focused Step-2 tests, `pip check` and
`git diff --check` passed. An independent CLI replay recorded zero training and
1,243 reference interactions; the scripted controller completed the generated
track in 948 steps with normalized progress approximately 1.0. The Phase-1
acceptance runner passed all 108 tests with only the existing Gymnasium warnings.

**Files**: `src/utils/metrics.py`, `src/utils/artifacts.py`,
`src/utils/references.py`, `src/utils/__init__.py`,
`experiments/evaluate_references.py`, focused tests, `README.md`, `.gitignore`,
`PLAN.md`, `docs/DIARY.md`.

**Commit**: `feature: add metrics artifacts and references [ai]`

## 2026-08-10 — Shared policy and value models

**Task**: Implement Phase-2 Step 3 with reusable neural components whose
capacity, initialization and probability semantics are controlled by the frozen
configuration.

**Result**: Added a generic Tanh MLP builder with orthogonal initialization from
a caller-owned PyTorch generator and no implicit global RNG consumption. Added
the bounded diagonal-Gaussian policy with state-independent learned dispersion,
reparametrized latent sampling, detached collection samples, deterministic mean
actions and the stable summed `tanh` change-of-variables log-probability. Added
the fixed-capacity scalar value network through the same builder and explicit
actor, critic and total trainable parameter counts. Model definitions remain
independent of racing observation dimensions.

**Review**: Verified the meta-device construction materializes ordinary
trainable parameters and initializes every real tensor from the supplied local
generator. Added coverage for clamped dispersion bounds and finite corrected
log-probabilities at extreme latent values.

**Validation**: Black, Ruff, Pyright, 11 focused model tests, `pip check` and
`git diff --check` passed. The Phase-1 acceptance runner passed all 119 tests;
Gymnasium retained only its existing advisory warnings.

**Files**: `src/models/mlp.py`, `src/models/policies.py`,
`src/models/value.py`, `src/models/__init__.py`, focused model tests, `PLAN.md`,
`docs/DIARY.md`.

**Commit**: `feature: add shared policy and value models [ai]`

## 2026-08-10 — Normalization, rollout records and fixed targets

**Task**: Implement Phase-2 Step 4 so exact network inputs and environment
boundaries survive collection and every learning target is verified before an
optimizer uses it.

**Result**: Added the documented float64 count/sum/squared-sum observation
normalizer with update-before-use training semantics, frozen bootstrap/evaluation
normalization, clipping, deterministic state restoration and checksums. Added an
immutable semantic on-policy transition containing normalized current/next
observations, latent and bounded actions, behaviour log-probability, current and
bootstrap values, separate termination/truncation flags, and episode/circuit
identity. Added complete-episode REINFORCE and fixed-rollout actor-critic buffers,
framework-ready detached tensor views, Monte Carlo return-to-go and fixed TD/GAE
targets. Truncation and rollout cuts bootstrap once; true termination does not;
GAE recursion ends at every environment or rollout boundary.

**Review**: After delegated implementation, retained the last finalized row so
the next rollout rejects duplicated or skipped transitions. Added one combined
hand-calculated fixture covering ordinary recursion, time-limit truncation, true
termination and a final rollout cut in the same batch.

**Validation**: Black, Ruff, Pyright, 21 focused buffer/normalizer tests,
`pip check` and `git diff --check` passed. The Phase-1 acceptance runner passed
all 140 tests with only the existing Gymnasium advisory warnings.

**Files**: `src/utils/normalizers.py`, `src/utils/buffers.py`,
`src/utils/__init__.py`, focused tests, `PLAN.md`, `docs/DIARY.md`.

**Commit**: `feature: add rollout buffers and learning targets [ai]`

## 2026-08-10 — Shared on-policy training, evaluation and resume engine

**Task**: Implement Phase-2 Step 5 by defining the project-owned agent boundary
and one collection, evaluation, accounting and checkpoint lifecycle for every
later algorithm.

**Result**: Added the continuous on-policy agent contract for stochastic
collection, deterministic actions, critic bootstrap, complete-episode or fixed-
rollout updates, diagnostics and agent-owned model/optimizer/generator state.
Added the shared racing training engine with exact interaction budgets,
update-before-use normalization, complete and fixed collection boundaries,
episode/update/evaluation counters and non-overlapping component timers. Added
fresh-environment deterministic evaluation at exact interaction boundaries with
frozen normalization. Added atomic versioned checkpoints containing agent,
normalizer, collector, active episode, focused environment, history, counters,
timing and named generator state, plus semantic RacingEnv/lifecycle snapshots for
mid-episode resume.

**Review**: After delegated implementation, added immutable engine/track/config
compatibility checks before restore, removed unnecessary process-global RNG
mutation, exercised an agent-owned stochastic generator, preserved earlier
episode/evaluation/update histories, added interaction-bound update events,
replaced access to buffer-private fields with restore APIs, validated restored
buffer structure, deep-copied environment RNG state, supported device-aware
checkpoint loading and normalized corrupt-checkpoint errors.

**Validation**: Black, Ruff, Pyright, 13 focused training/evaluation/checkpoint
tests, `pip check` and `git diff --check` passed. The resume fixture checkpoints
inside an active episode after prior updates and evaluations, then matches the
uninterrupted actions, local RNG state, summaries, counters and normalizer. The
Phase-1 acceptance runner passed all 153 tests with only the existing Gymnasium
advisory warnings.

**Files**: `src/agents/types.py`, `src/agents/__init__.py`,
`src/utils/training.py`, `src/utils/evaluation.py`,
`src/utils/checkpointing.py`, `src/utils/buffers.py`, `src/utils/__init__.py`,
`src/envs/racing/environment.py`, `src/envs/racing/lifecycle.py`,
`src/envs/racing/__init__.py`, focused tests, `PLAN.md`, `docs/DIARY.md`.

**Commit**: `feature: add shared on-policy training engine [ai]`

## 2026-08-10 — Resolve the Phase-2 code-review TODOs

**Task**: Apply every inline review request left after inspecting the Phase-2
configuration, model, recording, target and training-engine code.

**Result**: Simplified the configuration hierarchy by inlining one-field
collection and schedule objects, removing the learning-rate pair/calibration
wrappers, moving algorithm settings to `configs/algorithms.py`, and using enums
for algorithms and observation representations. Replaced the public seed and
integer-seed helpers with two generator operations in `utils/random.py`; the
protocol now uses sequential namespace identifiers and eight named streams,
including ordinary track generation. Reorganized run schemas and persistence
under `recording`, and collection, normalization, evaluation, checkpointing and
the engine under `training`; retained `utils` only for random and vector helpers.
Renamed artifact terminology to concrete run outputs and records.

Introduced the abstract `Policy` mapping alongside explicit scripted and random
policies. Split the learned role into an `ActorNetwork` that owns a documented
`GaussianPolicy`, replaced ambiguous latent names with `pre_squash_action`, made
the state-independent standard deviation a property, and documented the full
Gaussian sampling and corrected log-probability mathematics. Renamed the value
model and module to `CriticNetwork`/`critic.py`. Distinguished raw
`LoggedTransition` records from checkpointed `TrainingTransition` update data,
moved both to the recording schema, extracted shared vector conversions, and
removed redundant complete-episode integrity traversal. Renamed frozen
normalization to `normalize`, documented its running-sum equations, and clarified
why evaluation records retain the normalizer-state checksum.

**Validation**: Black, Ruff, Pyright, `pip check`, `git diff --check`, the full
153-test suite and the Phase-1 acceptance runner passed. Gymnasium retained only
its two existing advisory warnings about unbounded observation-space limits.
No unresolved TODO or stale moved-module reference remains.

**Files**: `src/configs/`, `src/models/`, `src/recording/`, `src/training/`,
`src/utils/`, `src/agents/types.py`, `experiments/evaluate_references.py`, focused
tests, `docs/LEARNING.md`, `docs/EXPERIMENT.md`, `README.md`, `PLAN.md`, and
`docs/DIARY.md`.

**Commit**: `refactor: resolve phase 2 review notes [ai]`

## 2026-08-10 — Actor-only REINFORCE learner

**Task**: Implement Phase-2 Step 6 with the exact complete-episode estimator,
controlled-task evidence and the first algorithm path through the shared runner.

**Result**: Added an actor-only `ReinforceAgent` that collects eight complete
episodes, computes Monte Carlo return-to-go, standardizes returns over the full
batch, averages trajectory-sum losses, recomputes corrected bounded-policy log
probabilities and applies the documented Adam update with global gradient-norm
clipping. Its checkpoint state owns the actor, optimizer and policy-sampling
generator. Diagnostics record loss, return and dispersion statistics, pre-clip
gradient norm, parameter norm and update magnitude.

Added the shared `experiments/train.py` entry point with explicit learning rate,
actor size, seed, device, interaction budget and run category. It uses the
configured evaluation and checkpoint boundaries, leaves partial Monte Carlo
batches unoptimized, records the actual deterministic execution policy, and
writes episode, update, evaluation, checkpoint and completion records. Added a
predeclared one-step positive-throttle problem and analytical, directional,
five-seed improvement, reproducibility and runner tests.

**Review**: After delegated implementation, made the serialized execution
device and evaluation cadence match runtime behaviour, configured PyTorch
determinism at runner startup, added intermediate checkpoint boundaries without
premature batch finalization, corrected persistence timing and extracted shared
parameter diagnostics for later actor-critic agents.

**Validation**: Six focused tests and all 159 tests passed. Black, Ruff, Pyright,
`pip check` and `git diff --check` passed; Gymnasium retained only its two
existing advisory warnings about infinite observation-space limits.

**Files**: `src/agents/reinforce.py`, `src/agents/diagnostics.py`,
`src/agents/__init__.py`, `experiments/train.py`, `tests/agents/`,
`tests/fixtures/envs/continuous_control.py`, `tests/experiments/test_train.py`,
`docs/LEARNING.md`, `README.md`, `PLAN.md`, and `docs/DIARY.md`.

**Commit**: `feature: implement REINFORCE agent [ai]`

## 2026-08-10 — A2C with detached GAE targets

**Task**: Implement Phase-2 Step 7 by adding the synchronous actor-critic
algorithm to the already validated model, target, engine and runner contracts.

**Result**: Added `A2CAgent` with a bounded Gaussian actor, fixed-capacity
V-function critic, separate Adam optimizers and fixed-rollout collection. Each
update computes the approved detached TD errors, raw GAE advantages and critic
targets; it standardizes advantages only for the mean actor loss and applies the
half-squared critic loss to fresh value predictions. Actor and critic gradients
are separately cleared, clipped and stepped. Checkpoints retain both models,
optimizers, configurations, learning rates and the policy-sampling generator.

Extended the shared training command and run-record writer for A2C without
adding another environment loop. Actor and critic initialization use independent
streams, critic counts and diagnostics are recorded, and the CLI requires the
critic learning rate only for actor-critic algorithms. Added an explicit
validation-only A2C gate plus independent loss reductions, gradient isolation,
the `lambda=0` target boundary, five-seed learning, reproducibility and racing
runner integration tests.

**Review**: After delegated implementation, made the generalized runner use the
existing `Algorithm` enum, strengthened the loss test so expectations do not
call the private loss methods, and centralized population standardization and
explained variance for identical use by REINFORCE, A2C and PPO. The acceptance
gate also exposed missing parameter-count typing in the generic runner; a
recording-specific extension of the agent protocol now declares those fields
without burdening engine-only test agents.

**Validation**: Twelve focused REINFORCE/A2C/runner tests and all 165 tests
passed. Black, Ruff, Pyright, `pip check` and `git diff --check` passed;
Gymnasium retained only its two existing infinite-bound advisory warnings.

**Files**: `src/agents/a2c.py`, `src/agents/diagnostics.py`,
`src/agents/reinforce.py`, `src/agents/__init__.py`, `experiments/train.py`,
`tests/agents/test_a2c.py`, `tests/experiments/test_train.py`,
`docs/LEARNING.md`, `PLAN.md`, and `docs/DIARY.md`.

**Commit**: `feature: implement A2C with GAE [ai]`

## 2026-08-10 — Clipped PPO with deterministic sample reuse

**Task**: Implement Phase-2 Step 8 by extending the fixed-rollout actor-critic
path with the approved clipped PPO objective and deterministic multi-epoch
minibatch reuse.

**Result**: Added `PPOAgent` with fixed detached behaviour log-probabilities,
GAE advantages and critic targets. Every update standardizes advantages once,
then uses a dedicated seeded permutation to cover every rollout row exactly once
per epoch. Each minibatch applies the clipped minimum actor objective and the
unclipped half-squared critic objective through separate Adam optimizers and
separate gradient-norm clipping. Checkpoints retain both models, optimizers,
policy-sampling state and minibatch-order state.

Extended the shared training command, run-record writer and CLI with PPO rather
than introducing a separate environment loop. PPO records actor and critic
losses, gradient, parameter and update norms, explained variance, sampled
entropy, importance-ratio statistics, clip fraction and the documented
nonnegative approximate-KL diagnostic. Unsupported optional objectives are
rejected when enabled because the approved configuration disables entropy
bonuses, value clipping, KL early stopping, weight decay and schedulers. Added
analytical clipping cases for both advantage signs, unit-ratio and zero-KL
checks, exact epoch coverage, fixed-old-policy evidence, five-seed controlled
learning, reproducibility and runner integration tests.

**Review**: After delegated implementation, checked the objective and target
boundaries directly against `docs/LEARNING.md` and the course notation, verified
that partial minibatches contribute sample-weighted diagnostics, and ran the
focused PPO, A2C, REINFORCE and shared-runner tests together before the full
repository gates.

**Validation**: Twenty focused cross-algorithm and runner tests and all 173
tests passed. Black, Ruff, Pyright, `pip check` and `git diff --check` passed;
Gymnasium retained only its two existing infinite-bound advisory warnings.

**Files**: `src/agents/ppo.py`, `src/agents/__init__.py`,
`experiments/train.py`, `tests/agents/test_ppo.py`,
`tests/experiments/test_train.py`, `docs/LEARNING.md`, `PLAN.md`, and
`docs/DIARY.md`.

**Commit**: `feature: implement clipped PPO [ai]`

## 2026-08-11 — Reproducible experiment analysis and reporting

**Task**: Implement Phase-2 Step 9 so every reported experiment result can be
reconstructed from run records, and resolve the inline code-review TODOs before
committing the work as one change.

**Result**: Extended schema-versioned run records with cumulative collection
and optimization times, resource and parameter counts, circuit geometry,
signal quantiles, value diagnostics, and deterministic trajectory checkpoints.
Reported runs require an explicit near-saturated steering threshold. Added a
deterministic analysis command that validates complete runs, inventories input
checksums, exports flattened and aggregated JSON/CSV tables, computes
seed-level learning AUC, convergence with right-censoring, final outcomes,
resource trade-offs, exact small-sample bootstrap intervals, identity-matched
paired differences, held-out circuit gaps and geometry strata. Added plots for
learning, outcomes, convergence/resources, optimization and
curvature-conditioned controls, retaining failed and incomplete episodes.

Resolved the review TODOs by giving every persisted record dataclass a
`Record` suffix, moving `TrainingTransition` into the buffer module, separating
`ActorNetwork` into `models/actor.py`, and replacing the sampled-action fields
with the policy-neutral `raw_action` and explicit `env_action`. Scripted,
random and Gaussian policies now all return `PolicySample`; mathematical code
docstrings use plain-text equations that render directly in editors. The
checkpoint schema was advanced because buffered transition field names changed.

**Review**: Verified that aggregation treats a root seed—not a checkpoint or
circuit—as the training replicate; that paired results join on declared
identities; that completion denominators and censored convergence remain
visible; and that every analysis output is regenerated only from the selected
recorded run directories. Synthetic fixtures cover known aggregates, ties,
failed roots, per-circuit pairing, trajectory selection and discovery-order
independence.

**Validation**: Black, Ruff, Pyright, `pip check`, `git diff --check`, the full
test suite and the Phase-1 acceptance runner passed. Gymnasium retained only
its two existing advisory warnings about unbounded observation-space limits.

**Files**: `src/recording/`, `src/training/`, `src/models/`, `src/agents/`,
`src/utils/analysis.py`, `src/utils/plotting.py`, `experiments/train.py`,
`experiments/analyze_results.py`, analysis and affected API tests, `README.md`,
`docs/EXPERIMENT.md`, `PLAN.md`, `requirements.txt`, and `docs/DIARY.md`.

**Commit**: `feature: add reproducible experiment analysis [ai]`

## 2026-08-11 — Explicit per-run actor learning rates

**Task**: Resolve the actor-learning-rate review TODO while retaining the
learning contract's requirement that calibration, rather than a preset, selects
the final rate.

**Result**: Added an optional learning rate to `ActorConfig`. Named actor-size
presets leave it unset, runners copy their required explicit command input into
a per-run actor configuration, and all three on-policy agents read the selected
rate from that configuration. Agent checkpoints were versioned because their
immutable actor configuration now includes the rate.

**Validation**: Focused configuration, agent and shared-runner tests passed;
Black, Ruff, Pyright and `git diff --check` passed.

**Files**: `src/configs/training.py`, `src/agents/reinforce.py`,
`src/agents/a2c.py`, `src/agents/ppo.py`, `experiments/train.py`, affected agent
and configuration tests, `docs/LEARNING.md`, and `docs/DIARY.md`.

**Commit**: `refactor: store actor learning rate in its config [ai]`

## 2026-08-11 — Readable algorithm-specific training engines

**Task**: Retain the reproducible shared training lifecycle while adding direct
algorithm-specific engines that make collection and update timing easier to
follow for educational use.

**Result**: Moved the unchanged shared engine to
`training/engines/shared_engine.py` and retained its public export for the
experiment runner. Added standalone REINFORCE, A2C and PPO engines whose main
methods show the complete agent-environment loops. REINFORCE keeps trajectories
separate and updates only after a full episode batch. A2C and PPO collect fixed
rollouts across episode boundaries and use a final shorter rollout. All three
support independently seeded circuit selection and reset streams, update the
observation normalizer only for current training inputs, and return concise
episode and optimizer-update records.

**Validation**: Added focused tests for complete-episode batching, full and
partial fixed rollouts, and deterministic multi-circuit selection. The full test
suite, Black, Ruff, Pyright, `pip check` and `git diff --check` passed.

**Files**: `src/training/engines/`, `src/training/__init__.py`,
`tests/training/test_algorithm_engines.py`, `PLAN.md`, `README.md`, and
`docs/DIARY.md`.

**Commit**: `feature: add educational training engines [ai]`

## 2026-08-11 — Guided REINFORCE racing notebook

**Task**: Present REINFORCE training as educational material using the
project-owned agent and its readable algorithm-specific engine, with visible
configuration, records, and rendered behavior.

**Result**: Added `notebooks/reinforce.ipynb`. It derives its explanation of
the bounded policy, complete-trajectory return-to-go, batch standardization and
trajectory-average loss from `docs/LEARNING.md`. One configuration cell exposes
the root seed, device, episode count, actor learning rate, REINFORCE settings,
render settings, and a switch between one explicit circuit seed and several.
The notebook constructs `ReinforceAgent` and `ReinforceTrainingEngine` directly,
renders deterministic rollouts before and after training, prints the collected
episode and update records, and plots return, progress, episode length and actor
loss. Jupyter was added as an explicit project dependency.

**Validation**: The notebook is valid version-4 JSON with stable cell IDs, every
code cell compiles, and reduced in-memory executions passed in both single- and
multi-circuit modes, including rendering and an optimizer update. The full test
suite, Black, Ruff, Pyright, `pip check` and `git diff --check` passed.

**Files**: `notebooks/reinforce.ipynb`,
`tests/notebooks/test_reinforce_notebook.py`, `requirements.txt`, `README.md`,
`PLAN.md`, and `docs/DIARY.md`.

**Commit**: `docs: add guided REINFORCE training notebook [ai]`

## 2026-08-11 — Live REINFORCE policy viewer

**Task**: Replace the notebook's static rollout-frame grid with a view that
shows the deterministic policy interacting with the racing environment action
by action.

**Result**: The REINFORCE notebook now offers an `inline` viewer that updates
one Jupyter figure after every environment action and a `window` fallback that
uses the environment's Pygame human renderer. The configuration cell exposes
the viewer mode, maximum visible steps and frame delay. Both the before- and
after-training views use this live interaction path while retaining the same
deterministic action and reset convention.

**Validation**: The inline viewer executed against a reduced training notebook,
and the complete notebook re-executed with its default configuration.

**Files**: `notebooks/reinforce.ipynb`,
`README.md`, and `docs/DIARY.md`.

**Commit**: `fix: add live notebook policy viewer [ai]`

## 2026-08-11 — Episode speed and throttle-magnitude records

**Task**: Show how the REINFORCE policy's mean vehicle speed and mean absolute
throttle/brake action change across training episodes.

**Result**: The readable REINFORCE engine now accumulates pre-action speed and
absolute longitudinal-action samples during each episode and stores their means
in `EducationalEpisodeRecord`. The notebook's records section exposes both
values and plots them alongside return, progress, episode length, and actor
loss.

**Validation**: The complete notebook executed with its current 640-episode CPU
configuration. It collected 1,003,399 training interactions, performed 80
optimizer updates, retained the new fields and six-panel plot, and finished
without cell errors.

**Files**: `src/training/engines/records.py`,
`src/training/engines/reinforce.py`, `notebooks/reinforce.ipynb`, and
`docs/DIARY.md`.

**Commit**: Pending user commit.

## 2026-08-11 — Educational A2C and PPO notebooks

**Task**: Add readable notebook walkthroughs for A2C with GAE and PPO, using
their algorithm-specific agents and engines rather than the shared experiment
engine.

**Result**: Added configurable A2C and PPO notebooks with explicit single- or
multi-circuit training, documented learning settings, raw episode returns, a
trailing 20-episode return mean, actor and critic weight norms at optimizer
steps, and algorithm-specific diagnostics. Both notebooks run isolated
deterministic evaluations before training and at a configurable episode
cadence, then show evaluation return, outcome and progress. The educational
A2C and PPO engines now expose an episode callback so evaluation does not force
an early fixed-rollout update, and both show episode progress bars.

**Validation**: The focused educational-engine test file passed all three tests,
and Black and Ruff passed on the touched Python files. Both four-episode smoke
notebooks executed and retained their outputs without cell errors. A2C collected
2,255 training interactions and two optimizer updates; PPO collected 2,353
training interactions and two optimizer updates. Each retained three greedy
evaluation records and its complete six-panel plot. No notebook unit tests were
added.

**Files**: `src/training/engines/a2c.py`,
`src/training/engines/ppo.py`, `tests/training/test_algorithm_engines.py`,
`notebooks/a2c.ipynb`, `notebooks/ppo.ipynb`, `README.md`, `PLAN.md`, and
`docs/DIARY.md`.

**Commit**: `feature: add educational A2C and PPO notebooks [ai]`

## 2026-08-11 — Aligned educational policy notebooks

**Task**: Bring the REINFORCE notebook's records and scheduled evaluation in
line with the A2C and PPO notebooks, and give the actor-critic notebooks the
same dynamic before/after policy view.

**Result**: REINFORCE now runs isolated deterministic evaluation before
training and after every configurable episode interval. Its episode callback
runs after any optimizer update triggered by the completed trajectory, so the
evaluation observes the updated actor at an eight-episode batch boundary. Its
records figure now combines raw and trailing 20-episode mean returns, greedy
evaluation return and progress, speed, throttle magnitude, actor loss and actor
weight norm. A2C and PPO now include the same inline live viewer and Pygame
fallback before and after training.

**Validation**: The focused educational-engine tests passed all three tests.
The REINFORCE notebook executed for 16 episodes, 11,041 interactions, two
updates and three greedy evaluations. The four-episode A2C and PPO notebooks
re-executed with three greedy evaluations each. All three retained their plots
and initial/final viewer output without cell errors. Black, Ruff, targeted
Pyright and `git diff --check` passed; no notebook unit tests were added.

**Files**: `src/training/engines/reinforce.py`,
`tests/training/test_algorithm_engines.py`, `notebooks/reinforce.ipynb`,
`notebooks/a2c.ipynb`, `notebooks/ppo.ipynb`, `README.md`, and
`docs/DIARY.md`.

**Commit**: `feature: align educational policy notebooks [ai]`

## 2026-08-11 — Shorter configurable training task

**Task**: Reduce episode wall time without changing the per-step reward formula,
agent timestep, discount, or learning equations.

**Result**: Reduced the configurable procedural-track scale from a `250m` to a
`50m` base radius and scaled its accepted length interval from `1000m`--`3000m`
to `200m`--`600m`. Generated development tracks are consequently about one fifth
as long. Reduced the configurable episode cap from 5000 to 1000 agent steps,
retaining the original little-more-than-two-times margin over the proportionally
shortened 450-step target lap. Restored the approved time-penalty rate to `0.05`.
The reward implementation remains unchanged and still divides signed progress
by the selected circuit's own length, so a full lap contributes approximately
one normalized progress unit at either scale.

**Validation**: Configuration and generation checks cover the new length and
time-limit values. Reward reference tests cover the `-0.9` target-lap step cost,
the `-2.0` stationary truncation total, the unchanged `-20` crash penalty, and
one normalized progress unit over a complete lap.

Saved-track validation derives its missing acceptance interval from the recorded
generation radius, so previously saved long circuits remain loadable without
weakening explicit generation-time validation.

**Files**: `src/configs/environment.py`, `src/envs/tracks/validation.py`,
`tests/configs/test_environment_config.py`, `tests/envs/test_track_generation.py`,
`tests/envs/test_reward.py`, `docs/MDP.md`, `docs/TRACK.md`, and `docs/DIARY.md`.

**Commit**: `config: shorten the training task [ai]`

## 2026-08-11 — Persistent deterministic racing workers

**Task**: Establish the process and random-stream foundation for concurrent CPU
environment collection without moving neural-network training to another device.

**Result**: Added indexed children beneath each existing named seed stream, so
every environment index owns reproducible policy-sampling, reset, and
track-selection identities without consuming a shared generator. Added a
persistent Gymnasium asynchronous vector environment that spawns its workers
once, steps enabled environments concurrently, parks inactive workers without
advancing their dynamics, and resets only a selected boundary mask. Every worker
applies the configured one-thread deterministic PyTorch policy for itself.
Vector state includes worker dynamics plus all scheduling-generator states for
later checkpoint integration. The default configured worker count now reflects
available physical CPU cores.

**Validation**: Focused tests verified indexed-stream independence, distinct and
stable worker process IDs across steps, worker-local deterministic PyTorch
settings, selective reset with parked environments, and exact vector-state
restore of the next transition.

**Files**: `src/utils/random.py`, `src/configs/training.py`,
`src/configs/__init__.py`, `src/training/vector_environment.py`,
`src/training/__init__.py`, `tests/utils/test_random.py`,
`tests/configs/test_training_config.py`, `tests/training/test_vector_environment.py`,
`requirements.txt`, `docs/LEARNING.md`, and `docs/DIARY.md`.

**Commit**: `feature: add persistent racing workers [ai]`

## 2026-08-11 — Short-circuit scripted baseline

**Task**: Preserve the deterministic scripted-policy environment check after
reducing the default circuit geometry to one fifth of its previous scale.

**Result**: Scaled the scripted controller's curvature feed-forward gain from
`50` to `10` with the circuit geometry. This controller is a non-learning
reference and does not change the reward formula or any learning equation. It
now completes the deterministic shortened baseline circuit within the new
episode cap.

**Validation**: The formula reference test was updated for the scaled gain, and
the end-to-end scripted baseline completed one lap in 306 interactions.

**Files**: `src/models/policies.py`,
`tests/training/test_policy_evaluation.py`, and `docs/DIARY.md`.

**Commit**: `fix: scale scripted baseline for short circuit [ai]`

## 2026-08-11 — Parallel on-policy rollout collection

**Task**: Reduce training wall time by collecting racing interactions in
persistent CPU processes while preserving the reward, learning equations,
episode-boundary rules, deterministic seed contract and disabled features.

**Result**: Batched actor and critic inference now consumes synchronous
observation rows without moving model training to another device. REINFORCE
uses eight worker-indexed policy, reset and circuit-selection streams and
collects one complete trajectory per worker concurrently before its unchanged
eight-trajectory update. A2C and PPO default to the number of physical CPU
cores and fill an explicit `(time, environment, ...)` buffer to exactly 2048
pooled valid transitions. Environments reset independently after termination or
truncation, and GAE recurses backward within each environment column only.

Both the complete experiment engine and the readable algorithm engines spawn
their process pool once and reuse it for the run. Checkpoints now retain every
worker, observation, collector and indexed generator state. The experiment
runner exposes `--num-envs`, records worker stream identities, and uses eight
workers automatically for REINFORCE. Optional collision metadata is encoded
homogeneously at the Gymnasium vector boundary and decoded back to its semantic
`None` value, preventing an asymmetric worker crash from blocking collection.

The learning and experiment documents describe the parallel collection shape
and execution contract. All three educational notebooks expose their worker
count, construct indexed streams, close their workers explicitly and retain
16-episode smoke outputs. No entropy bonus, learning-rate scheduler, weight
decay, PPO value clipping, KL early stopping or GPU migration was added;
`delta_t_agent=0.04`, `gamma=0.9995` and the `0.05` time-penalty rate remain
unchanged.

**Validation**: Hand-computed vector-buffer tests verify the valid mask and
independent GAE columns. Process tests verify persistent worker identities,
selective reset, deterministic worker settings, state restore and asymmetric
optional collision data. The focused policy/agent/buffer/engine/runner suite
passed 69 tests before the final integration pass. Black, Ruff and targeted
Pyright passed, and the complete suite passed all 198 tests with only the two
existing unbounded-observation-space warnings.

The retained notebooks completed without cell errors. REINFORCE recorded 16
episodes, 8,674 interactions, two updates and three deterministic evaluations;
A2C recorded 16 episodes, 8,584 interactions, five updates and nine evaluations;
PPO recorded 16 episodes, 10,679 interactions, six updates and nine evaluations.
Each retains its initial/final live viewer and training-record plots. No
notebook unit tests were added.

**Files**: `src/models/actor.py`, `src/models/policies.py`, `src/agents/`,
`src/training/buffers.py`, `src/training/normalization.py`,
`src/training/vector_environment.py`, `src/training/engines/`,
`experiments/train.py`, `tests/models/test_policies.py`, `tests/training/`,
`tests/experiments/test_train.py`, `notebooks/`, `docs/LEARNING.md`,
`docs/EXPERIMENT.md`, `README.md`, `PLAN.md`, and `docs/DIARY.md`.

**Commit**: `feature: parallelize on-policy rollouts [ai]`

## 2026-08-11 — Smoothed REINFORCE diagnostics and exploration scale

**Task**: Make the noisy REINFORCE training records easier to inspect and show
the actor's learned Gaussian exploration parameter.

**Result**: The REINFORCE notebook now overlays trailing moving averages with
window 20 on mean episode speed, mean episode throttle magnitude, and actor loss
per optimizer update, in addition to the existing return average. Its records
figure is organized as a 3-by-3 grid and retains all previous metrics. A new
panel shows the separate throttle and steering exploration scales computed from
the recorded learned log standard deviations using
`sigma = exp(log_standard_deviation)`, as specified in `docs/LEARNING.md`.

**Validation**: The notebook's JSON and every Python cell parsed successfully.
The retained 16-episode execution completed without cell errors and produced
the revised plot from 8,674 interactions, two optimizer updates and three
deterministic evaluations. Both sigma series contain one value per update.

**Files**: `notebooks/reinforce.ipynb` and `docs/DIARY.md`.

**Commit**: `docs: extend REINFORCE training plots [ai]`

## 2026-08-11 — Optional REINFORCE actor-weight decay

**Task**: Add an experimental actor-weight penalty for investigating the
non-learning REINFORCE notebook policy without directly regularizing its
exploration parameter.

**Result**: Added the explicit `actor_weight_decay` coefficient to
`ReinforceConfig`, defaulting to zero so existing runs retain their previous
optimizer. The REINFORCE Adam optimizer now has two clear parameter groups:
MLP weight matrices receive the configured L2 penalty, while biases and the
learned throttle/steering log standard deviations receive zero weight decay.
The agent records the coefficient with each optimizer update and rejects
negative values. Its checkpoint state version was advanced because the
optimizer parameter-group structure changed.

The educational REINFORCE notebook enables `actor_weight_decay=1e-4` and
explains its scope. Its existing long-run episode, batch, evaluation and reward
edits were preserved, and the notebook was deliberately not re-executed.
A2C and PPO continue to disable weight decay.

**Validation**: Focused agent, configuration, educational-engine, shared-engine
and experiment-runner tests passed all 29 tests. A new optimizer-structure test
proves that every MLP `weight` parameter is in the `1e-4` group and that every
bias plus the learned log standard deviation is in the zero-decay group. A
second test covers rejection of negative coefficients. The notebook remains
valid JSON and every Python cell parses successfully.

**Files**: `src/configs/algorithms.py`, `src/agents/reinforce.py`,
`tests/agents/test_reinforce.py`, `notebooks/reinforce.ipynb`,
`docs/LEARNING.md`, and `docs/DIARY.md`.

**Commit**: `feature: add REINFORCE actor weight decay [ai]`

## 2026-08-11 — Reward rebalance: why all three algorithms learned to stall

**Task**: Diagnose why REINFORCE, A2C and PPO all converge to a stalled policy
on the fixed development circuit, and fix the shared cause rather than tuning
each algorithm.

**Result**: The cause is the reward coefficients, not the learning code. The
implementations of the score-function estimator, the tanh change-of-variables
correction, the GAE recursion, the truncation-versus-termination bootstrap and
the seed contract were all checked against `docs/LEARNING.md` and are correct.
The return landscape they were asked to climb was not.

Measured on the seed-0 circuit under the previous coefficients
($R_{\text{finish}}=10$, $R_{\text{crash}}=20$, $c_{\text{prog}}=1$):

| behaviour | lap fraction | return |
|---|---|---|
| never leave the start line | 0.000 | **-0.30** |
| drive, crash immediately | 0.110 | -19.91 |
| drive, crash after a third of the lap | 0.400 | -19.65 |
| complete the lap | 1.001 | +10.91 |

The whole competence ladder from "cannot steer" to "drives a third of the lap"
spans 0.26 return units, and all of it sits 19 units *below* doing nothing. Two
orderings were inverted at once:

1. $R_{\text{crash}}=20$ exceeded the $c_{\text{step}}T_{\max}=0.3$ cost of
   idling to the time limit, so standing still strictly dominated every attempt
   to drive. Taking the measured partial-lap and full-lap returns, a policy had
   to already complete **63%** of its laps before attempting the lap beat
   standing still in expectation — unreachable from a 0% completion rate.
2. $c_{\text{prog}}=1$ against $R_{\text{crash}}=20$ made the dense shaping term
   worth 5% of one terminal event, so the return carried almost no information
   about *where* an episode ended, only *whether* it crashed.

The start state makes the resulting attractor absorbing. The car starts at
$v_0=0$, and the kinematic kernel freezes pose and heading at zero speed, so a
policy with negative mean throttle receives a constant observation and a
constant reward forever. Sampling 16 rollouts from the documented initial policy,
the standardized Monte Carlo return correlates $-0.31$ with instantaneous speed
(a slope of $-0.21$ standard deviations per m/s): the estimator was explicitly
telling every algorithm to slow down. A2C and PPO reach the same place through a
critic that learns $v(o)\approx-20\gamma^{\text{steps to crash}}$.

New coefficients are $R_{\text{finish}}=100$, $R_{\text{crash}}=5$,
$\rho=0.5s^{-1}$ and $c_{\text{prog}}=100$. Both inequalities now hold
($5 < 20$ and $100 \gg 5$). Idling to truncation becomes the worst available
outcome at $-20$; crashing after a third of a lap returns $+31$; a complete lap
returns $+194$. The return is monotone in distance covered and decreasing in
time taken, which is the shortest-time objective the project set out to encode.
The same rollout sample now shows a speed correlation of $-0.03$, and return
gained per unit lap fraction rises from $0.90$ to $92.81$.

Nothing else changed: the reward *formula*, the observation, the action mapping,
$\gamma=0.9995$, $\Delta_{t_{agent}}=0.04$, the episode cap, the physics version,
the seed contract and every learning equation are untouched. Only the four
`RewardConfig` scalars moved. This also restores the reward reference tests,
which the previous ad-hoc `0.05 -> 0.0075` time-penalty edit had left failing.

**Validation**: Two new invariant tests replace the previous fixed-total checks
as the real regression guard: crashing anywhere on the circuit must outscore
idling to the time limit, and driving further before crashing must score
strictly higher. A third asserts the two coefficient inequalities directly, so a
future retune cannot silently reintroduce the inversion. Reference totals were
updated to $-9$ over a target lap, $-20$ for a stationary timeout, $+100$ for a
finish and $-5$ for a crash.

End-to-end, the algorithms were retrained headlessly on the seed-0 circuit
across eight workers. The comparison is controlled: identical actor and critic
architectures, learning rates, $\gamma$, $\lambda$, gradient-norm limit, seed
namespace and worker count. Only the four reward scalars differ.

PPO, in 64-episode buckets, mean lap fraction and outcome mix:

| bucket | previous coefficients | new coefficients |
|---|---|---|
| 64 | 0.076, 80% time limit | 0.414, 11% laps completed |
| 128 | 0.067, 86% time limit | 0.956, 84% laps completed |
| 192 | 0.008, **100% time limit** | 0.991, **98% laps completed** |

At 192 episodes the new run averages 258 agent steps per episode, below the 306
steps the scripted Frenet reference needs for the same circuit. The previous
coefficients produce a permanent stall, matching `notebooks/ppo.ipynb`.

A2C over 384 episodes and 220,039 interactions climbs monotonically and never
stalls: mean lap fraction 0.127, 0.180, 0.214, 0.279, 0.290, 0.340 with mean
return -3.13, 1.93, 5.21, 11.67, 13.00, 18.76 and a best episode reaching 0.686
of the lap. A2C applies one gradient step per 2,048-transition rollout, so it is
expected to trail PPO's ten epochs per rollout. For reference,
`notebooks/a2c.ipynb` finished 844,000 interactions at lap fraction 0.030 with
every episode ending at the time limit.

REINFORCE applies one update per eight complete episodes, so over 512 episodes
and 290,184 interactions it receives only 64 gradient steps and improves
slowest. Its mean lap fraction still rises across every one of its eight
buckets:

| episodes | 64 | 128 | 192 | 256 | 320 | 384 | 448 | 512 |
|---|---|---|---|---|---|---|---|---|
| mean lap fraction | 0.118 | 0.124 | 0.146 | 0.163 | 0.178 | 0.208 | 0.229 | 0.259 |
| mean return | -3.87 | -3.18 | -1.57 | 0.61 | 1.28 | 4.14 | 6.85 | 9.24 |
| time limit % | 0 | 0 | 8 | 6 | 5 | 5 | 2 | 14 |

Return crosses zero by the 256th episode and the best episode reaches 0.449 of
the lap. The time-limit share never approaches the 100% recorded in
`notebooks/reinforce.ipynb` after 1,325,384 interactions, and those time limits
are now slow laps in progress rather than a stalled car, since idling is the
worst-scoring outcome available. The stall attractor is removed for REINFORCE as
well; its remaining slowness is the sample efficiency of a Monte Carlo estimator
given 64 updates, not the reward pathology.

The claim being validated is specifically that the stall is no longer an
attractor for any of the three, and that return now increases with distance
covered. Reaching a reported-experiment level of performance for A2C and
REINFORCE needs the full interaction budget, not these short diagnostic runs.

The complete suite passes 201 tests. The two `tests/experiments/test_train.py`
failures present afterwards were verified to fail identically on `main` before
this change; they depend on the host's physical core count and are unrelated.
Black and Ruff pass. The three notebooks were deliberately not re-executed.

**Files**: `src/configs/environment.py`, `tests/envs/test_reward.py`,
`tests/envs/test_episode_lifecycle.py`,
`tests/configs/test_environment_config.py`, `docs/MDP.md`, and `docs/DIARY.md`.

**Commit**: `fix: rebalance reward coefficients [ai]`

## 2026-08-12 — Interaction-budgeted educational training

**Task**: Give the three readable notebook engines the same exact environment-
interaction stopping rule and complete the common episode summaries needed for
aligned result plots.

**Result**: Replaced episode-count stopping in the REINFORCE, A2C and PPO
educational engines with an absolute interaction budget. Vector collection now
limits the active worker columns at the final boundary, so the counter ends at
the requested value even when it is not divisible by the worker count. A2C and
PPO optimize their final short rollout; REINFORCE retains completed episode
records but does not update from an incomplete Monte Carlo batch, matching the
learning contract. A2C and PPO now record the same per-episode mean speed and
mean throttle magnitude already exposed by REINFORCE.

**Validation**: The focused educational-engine and vector-environment suite
passed all eight tests. Black and Ruff passed on the changed engine, record and
test files.

**Files**: `src/training/engines/records.py`,
`src/training/engines/reinforce.py`, `src/training/engines/a2c.py`,
`src/training/engines/ppo.py`, `tests/training/test_algorithm_engines.py`, and
`docs/DIARY.md`.

**Commit**: `feature: budget educational training by interactions [ai]`

## 2026-08-12 — Episode-indexed procedural training circuits

**Task**: Replace the educational notebooks' three hand-picked training
circuits with a reproducible fresh procedural circuit at every episode reset.

**Result**: The persistent vector workers can now replace their fixed-pool
circuit with a procedurally generated circuit from an explicit seed during
reset, while retaining fixed-pool behaviour for single-circuit runs. The
readable engines accept an episode-to-seed function and request the circuit for
the logical episode identity before resetting a worker. This makes circuit
identity independent of worker completion order and allows REINFORCE, A2C and
PPO to share the same root-and-episode schedule. Worker snapshots retain both
fixed track indexes and procedural seeds for exact reconstruction.

**Validation**: Vector tests verified explicit procedural identities and saved
worker seeds. The educational-engine test verified that episode identities
`0..2` received procedural seeds `100..102`. The focused engine and vector
suite passed all eight tests; Black and Ruff passed.

**Files**: `src/training/vector_environment.py`,
`src/training/engines/reinforce.py`, `src/training/engines/a2c.py`,
`src/training/engines/ppo.py`, `tests/training/test_vector_environment.py`,
`tests/training/test_algorithm_engines.py`, `README.md`, and `docs/DIARY.md`.

**Commit**: `feature: generate a circuit for each training episode [ai]`

## 2026-08-12 — Aligned algorithm explanations and result dashboards

**Task**: Perfect the REINFORCE, A2C and PPO notebooks with the same narrative,
configurable before/after rendering, common statistics and categorized result
presentation.

**Result**: Rebuilt all three notebooks around the same progression: full
algorithm motivation and project equations, common interaction-budget and
procedural-circuit configuration, short initial dashboard, scheduled greedy
evaluation, training, four common result categories, algorithm-specific
diagnostics where applicable, and a full-length final dashboard. The A2C
description now motivates the critic and GAE from REINFORCE's variance and
delayed updates; PPO motivates clipped sample reuse from A2C's one-update-per-
rollout limitation.

Added one shared educational visualization module. Its live deterministic
viewer takes `max_steps` explicitly, combines the RGB circuit with current
progress, return, speed, throttle and steering, and plots speed plus both
controls over time. Notebook calls use 20 actions before training and the
configured 1000-action episode limit afterwards. Its common figures separate
task performance, progress/episode efficiency, driving behaviour, and
optimization/exploration. They show every requested moving average, greedy
return mean plus/minus one standard deviation, actor/critic norms, and the
learned throttle/steering sigmas as the policy-gradient exploration trace.
A2C and PPO retain critic diagnostics, and PPO retains clip fraction and
approximate KL.

**Validation**: Black and Ruff passed, targeted Pyright reported no issues, and
the complete project suite passed all 204 tests. All notebook JSON and Python
cells parsed. Per the user's request, no notebook-focused test file was added.
Instead, reduced in-memory Jupyter executions ran every code cell in REINFORCE,
A2C and PPO without errors while leaving the committed 2,000,000-interaction
configurations and output-free notebooks unchanged. Separate runtime smokes
exercised the live dashboard and every shared figure builder.

**Files**: `src/training/educational_visualization.py`,
`notebooks/reinforce.ipynb`, `notebooks/a2c.ipynb`, `notebooks/ppo.ipynb`,
`README.md`, and `docs/DIARY.md`.

**Commit**: `feature: align algorithm notebook dashboards [ai]`

## 2026-08-12 — Environment step made three times faster

**Task**: Reduce the cost of one `RacingEnv.step`. Profiling the 2,000,000
interaction runs showed the environment, not the learning code, setting the
wall-clock ceiling for Experiment 1.

**Result**: Three changes, none of which alter behaviour. Centerline projection
now evaluates all candidate segments in one vectorized pass instead of a Python
loop calling `project_to_segment` per segment; `project_sorted_candidates` is
the documented unvalidated hot path and the Frenet observer precomputes the
sorted local-candidate table once per track instead of building a Python set per
substep. SciPy's `CubicSpline` objects are still used to build coefficients, but
scalar evaluation goes through `ScalarPiecewisePolynomial`, which applies
Horner's rule directly and skips SciPy's array dispatch; position now uses a
single vector-valued spline instead of two scalar ones. The finish-gate geometry
is resolved once per episode into a `FinishGate` rather than re-interpolated at
every physics substep. `RacingEnv._observe` also passes the lifecycle's known
segment index, removing one global KD-tree search per agent step.

Measured on this machine: `1164 us` to `348 us` per step, a factor of `3.34`.

This exposed a second problem that is left for a later decision. With the step
this cheap, the spawned-worker `AsyncVectorEnv` no longer pays for itself:
ten workers deliver `2,605` steps per second while ten environments stepped
sequentially in one process deliver `2,821`. Pure inter-process overhead is
`1.01 ms` per wave against a `0.35 ms` step. Exploiting more than one core now
needs a batched environment that steps every car with array operations, not
per-step process handoff.

**Validation**: A scripted-policy trace over three circuits and 926 steps,
covering a crash, a completed lap and a time limit, is bit-identical before and
after. The environment and geometry suites pass; Black and Ruff pass.

**Files**: `src/envs/geometry/interpolation.py`,
`src/envs/geometry/projection.py`, `src/envs/geometry/__init__.py`,
`src/envs/observations/frenet.py`, `src/envs/racing/lifecycle.py`,
`src/envs/racing/environment.py`, `src/envs/tracks/track.py`, and
`docs/DIARY.md`.

**Commit**: `perf: vectorize centerline projection and spline evaluation [ai]`

## 2026-08-12 — Grip-limited physics, sampled starts and a lap-time reward

**Task**: Act on the review of the 2,000,000-interaction runs recorded in
`RESPONSE.md`. Three symptoms shared one theme: the environment did not ask the
policy for anything a racing driver would recognize. Full throttle was optimal
everywhere, standing still was reachable by gradient ascent from a neutral
initialization, and a lap three times slower scored within five percent of a
fast one.

**Result**: Four changes to the environment and two to the learning contract.

*Grip.* Longitudinal and lateral tyre demand now share one friction budget of
`20 m/s^2`, about `2g`, chosen to match the lateral acceleration limit the
scripted reference controller already assumed. Asking for more cornering than
the budget leaves does not spin the car or trigger a special rule: the achieved
steering angle is reduced, the car understeers, runs wide and eventually reaches
the track boundary. Crashing therefore remains a geometric event. Braking uses
the full budget while acceleration stays engine-limited at `9.26 m/s^2`.
Quadratic drag, derived as `a_max / v_max^2` rather than introduced as a free
constant, makes `70 m/s` a terminal speed instead of a clamp.

*Steering rate.* The front wheels travel at most `180°/s`, so a full sweep takes
a third of a second rather than one agent step. This makes the steering angle a
state variable, so the Frenet observation gains it and becomes five-dimensional.
Hiding it would have made the task partially observable for no modelling reason.

*Start states.* Training samples the start pose uniformly along the circuit with
bounded lateral offset, heading error and speed. The finish gate moves with the
start, so a lap is always one full circuit. Deterministic evaluation still
launches from the canonical start line, so reported numbers stay comparable and
one evaluation episode per checkpoint is now genuinely sufficient.

*Stall rule.* A car that advances less than one metre in three seconds ends its
episode, charged the entire time penalty it would have paid by idling to the
limit. The return of standing still is unchanged; only the simulation is saved.
It also gives the failure its own recorded outcome instead of hiding it as a
time limit.

*Reward.* The per-step time penalty doubles to `1 s^-1` and the completion
reward gains a term scaled by the unused episode clock. Raising the step penalty
alone could not carry lap time: past roughly `2 s^-1` a policy that drives half
a lap and crashes scores below one that crashes immediately, which re-inverts
the ordering fixed the previous day. A term that applies only on success has no
such side effect. `experiments/plot_reward.py` draws all three orderings beside
the original coefficients, for which every one of them was violated.

*Learning contract.* The learned exploration scale is capped at `1` instead of
`exp(2)`. PPO's runs showed it growing `0.61 -> 1.9`, which with `tanh` squashing
saturates nearly every sample and turns the policy into random bang-bang
steering. PPO now reuses each rollout four times instead of ten and stops early
on approximate KL. REINFORCE loses its actor weight decay, which no other
algorithm had and which shrank the actor monotonically toward a zero-output mean.

**A regression this introduced, and its fix**: the first validation run showed
A2C stalling in 97% of training episodes, worse than before the change. The
cause was the braking asymmetry interacting with symmetric exploration noise: at
a zero mean action, `E[acceleration] = 9.26 E[tanh^+] - 20 E[tanh^-] =
-2.17 m/s^2`. An untrained policy brakes to a standstill within seconds. A
control run with symmetric braking confirmed it, dropping stalls to under five
percent. The fix keeps the physics and corrects the initialization instead: the
actor's throttle output bias starts at `0.2`, the value at which the initial
policy's expected acceleration is zero. Stalls fell to `0-9%` and the mean
signed throttle moved from `0.00` to `0.16`. A policy that is neutral in the
action is not neutral in the quantity the action controls.

**Validation**: The scripted reference controller, retuned for the new physics
(target lateral acceleration `12 m/s^2`, below the budget, because preview
curvature understates a corner on entry and braking spends grip that cannot then
be used to turn), completes 24 of 24 generated circuits averaging `15.9 s`. This
is the standing evidence that the task remains solvable. A 926-step scripted
trace over three circuits is bit-identical before and after the projection
optimization. The full suite passes, including two `test_train.py` failures that
predated this work and depended on the host's core count.

**Files**: `src/configs/environment.py`, `src/configs/training.py`,
`src/configs/algorithms.py`, `src/envs/vehicle/`, `src/envs/observations/frenet.py`,
`src/envs/racing/`, `src/models/policies.py`, `src/agents/ppo.py`,
`src/agents/reinforce.py`, `src/training/`, `experiments/plot_reward.py`,
`notebooks/`, `docs/MDP.md`, `docs/TRACK.md`, `docs/EXPERIMENT.md`, `README.md`,
and `docs/DIARY.md`.

**Commits**: `feature: add grip-limited physics, sampled starts and lap-time
reward [ai]`, `feature: bound exploration scale and shorten PPO rollout reuse
[ai]`, `feature: report signed throttle, outcome mix and lap time [ai]`,
`fix: initialize the policy neutral in acceleration, not in action [ai]`,
`docs: specify the grip-limited MDP and the three reward orderings [ai]`.
