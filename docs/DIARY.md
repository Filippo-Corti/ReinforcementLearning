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
