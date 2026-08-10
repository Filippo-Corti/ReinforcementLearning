# Project Plan — Phase 2: Experiment-Ready Learning System

## Goal

Build and validate every environment, learning, reporting and orchestration
component needed to run the two studies specified in
[`docs/EXPERIMENT.md`](docs/EXPERIMENT.md).
At the end of Phase 2, the repository must be able to launch the complete
experiment matrices without adding code or making an undocumented scientific
choice.

Phase 2 must:

- establish deterministic configuration, seeding, logging, checkpointing and
  evaluation contracts;
- implement a shared configurable policy MLP and the value-function components
  needed by actor-critic algorithms;
- implement project-owned REINFORCE, A2C with GAE and PPO from approved
  equations;
- prove that every algorithm learns on a small controlled problem and that the
  complete racing path runs end to end;
- evaluate a capable policy on version-0 physics before deciding whether to add
  lateral grip;
- add and validate the minimum grip constraint only if the pre-experiment
  capability diagnosis confirms unrealistic high-speed cornering;
- add LiDAR observations without duplicating the racing dynamics or lifecycle;
- support deterministic training on procedurally generated circuits and
  evaluation on disjoint held-out circuits;
- generate all raw records and derived summaries required by both experiments;
  and
- run reduced-budget end-to-end validation on every Experiment 1 and Experiment
  2 configuration before the expensive reported experiment runs begin.

The full multi-seed reported experiment runs are not Phase-2 acceptance tests. They
begin only after this roadmap is complete and the experiment manifest has been
frozen. Pre-experiment configuration and reduced-budget validation results must
use seed namespaces and result directories that cannot be mistaken for reported
experiment results.

## Scientific Boundary

[`docs/EXPERIMENT.md`](docs/EXPERIMENT.md) is the authoritative study design. This file
describes how to build and validate the system that will execute that design.
The distinction is important:

- **Phase-2 validation** asks whether implementations are correct,
  deterministic and connected properly.
- **Pre-experiment configuration work** chooses or verifies predeclared settings
  without being reported as final evidence.
- **Reported experiment runs** estimate the effects of policy-network size,
  learning algorithm and observation representation using the frozen protocol.

Pre-experiment results may change a setting only through an explicit update to
`docs/EXPERIMENT.md` made before reported runs start. Once the reported-run
manifest is frozen, failed or surprising runs are retained; they are not
repaired by changing budgets, metrics, reward, tracks or hyperparameters for
selected configurations.

## Authoritative Specifications

Implementation decisions must agree with:

- [`docs/EXPERIMENT.md`](docs/EXPERIMENT.md) for hypotheses, comparison matrices,
  experimental units, measures, aggregation and fairness rules;
- [`docs/MDP.md`](docs/MDP.md) for state, observations, actions, dynamics,
  reward and episode lifecycle;
- [`docs/TRACK.md`](docs/TRACK.md) for track generation, geometry, Frenet
  projection, LiDAR ray casting, progress and lap completion;
- the learning specification introduced in Step 0 for project-specific policy,
  target and loss equations;
- the relevant derivations in `docs/theory/`, in particular
  `policy-gradient-1.md`, `policy-gradient-2.md`, `actor-critic.md` and
  `deep-rl.md`;
- [`AGENTS.md`](AGENTS.md) for workflow, code organization, determinism,
  dependency and documentation rules; and
- the archived Phase-1 roadmap in
  [`docs/old-plans/phase-1-racing-environment-mvp.md`](docs/old-plans/phase-1-racing-environment-mvp.md)
  for the accepted environment baseline.

The theory notes establish the algorithm family, but do not settle every
implementation detail. `docs/LEARNING.md` must resolve truncation bootstrap,
loss reduction, action bounding, normalization and optimizer semantics before
agent code is written.

## Experimental Invariants

The following invariants apply across the implementation and experiment
configuration:

1. The Experiment 1 size factor changes the **actor/policy network only**. The
   critic architecture remains fixed across sizes so critic capacity is not
   confounded with policy-space complexity.
2. Small, medium and large actors use hidden sizes `(32, 32)`, `(64, 64)` and
   `(256, 256)` respectively. Actor parameter count is recorded rather than
   inferred only from these labels.
3. Every Experiment 1 cell receives the same training-environment interaction
   budget, evaluation cadence, fixed circuit, observation, reward, physics
   version and five paired root seeds.
4. Algorithm-specific hyperparameters may differ because the algorithms have
   different update rules. They are calibrated only with the medium actor and
   dedicated pre-experiment roots, then frozen before the size sweep.
5. Experiment 2 uses PPO and selects one actor size by the predeclared rule in
   `docs/EXPERIMENT.md`; it does not choose the most flattering network after looking
   at held-out circuits.
6. Frenet and LiDAR runs use paired root seeds, training-track schedules,
   budgets, PPO settings and held-out circuits. Neither actor nor critic receives
   privileged observations in the LiDAR condition.
7. Training, evaluation and pre-experiment interactions are counted and stored
   separately.
8. Deterministic evaluation never updates model parameters, optimizer state,
   normalization statistics or random streams used for training.
9. A completed lap and a crash are environment outcomes, not inferred from a
   reward threshold.
10. Return is reported, but task conclusions also use completion, lap time,
    progress and crash rate so reward shaping cannot stand in for driving
    performance.

## Scope Boundaries

Phase 2 includes the complete software and validation path for both experiments,
including short pre-experiment configuration runs and reduced-budget end-to-end
validation runs. It excludes:

- the full 45-run Experiment 1 reported-run matrix;
- the full paired Experiment 2 reported-run matrix;
- final hypothesis tests or scientific conclusions from those runs;
- changing the reward after individual reported results are observed;
- recurrent policies or frame stacking for the LiDAR condition;
- finite vehicle footprint, tire slip, aerodynamic drag, load transfer and
  steering-rate limits;
- broad or per-network hyperparameter searches;
- selecting tracks or checkpoints using held-out test performance;
- comparing additional algorithms or observation types; and
- treating pre-experiment configuration or reduced-budget validation results as
  reported experiment data.

Randomized start states, reward scaling, entropy bonuses, gradient clipping and
other training mechanisms are not assumed silently. If the approved learning
contract uses one, its value and scope must be explicit and it must be held
fixed wherever `docs/EXPERIMENT.md` requires.

## Definition of Done

Phase 2 is complete only when all of the following are true:

- `docs/LEARNING.md` contains every equation and boundary convention needed to
  implement REINFORCE, A2C+GAE and PPO for bounded, vector-valued actions.
- The experiment document contains no unresolved choice that can alter a
  reported result: budgets, evaluation cadence, convergence threshold,
  algorithm settings, fixed circuit, seed sets, track splits and selection
  rules are frozen in a machine-readable manifest.
- One root seed deterministically derives independent streams for parameter
  initialization, policy sampling, environment reset, training-track selection,
  minibatch order and evaluation.
- Configurations, dependency versions, git state, machine context and complete
  seed provenance are retained with every run.
- The shared MLP, bounded Gaussian policy, value network and observation
  normalizer pass shape, gradient, numerical and reproducibility tests.
- Rollout records preserve observation, action, reward, log-probability, value,
  termination, truncation, next observation, episode and track identity.
- Hand-computed trajectories verify Monte Carlo return-to-go, bootstrapped
  returns and GAE under both termination and truncation.
- REINFORCE, A2C+GAE and PPO match analytical loss cases and improve a small
  deterministic continuous-control task across the approved validation seeds.
- The shared racing runner can train, checkpoint, resume and deterministically
  evaluate all three algorithms without algorithm-specific experiment scripts.
- Episode, update, evaluation, timing, resource and optimization diagnostics
  use one documented schema and can be aggregated without parsing console text.
- A capable version-0 policy has been evaluated by curvature, speed, throttle
  and steering, and the decision to retain or replace version-0 physics is
  recorded before experiment manifests are frozen.
- If grip is required, its equations and constants are approved in
  `docs/MDP.md`, version-0 behaviour remains selectable, and analytical and
  behavioural validation passes without changing the reward.
- Both Frenet and LiDAR environments pass Gymnasium conformance and deterministic
  observation tests while sharing the same dynamics and lifecycle.
- A deterministic multi-track scheduler produces disjoint development,
  training, validation and test track streams without regenerating a different
  circuit after resume.
- A reduced Experiment 1 matrix executes every algorithm/actor-size cell and
  produces the expected artifacts.
- A reduced Experiment 2 matrix trains and evaluates paired Frenet and LiDAR
  PPO runs on disjoint procedurally generated circuits.
- The analysis command regenerates tables and plots from raw artifacts and does
  not depend on manually copied values.
- The final acceptance runner passes dependency, formatting, linting, type,
  compilation, test, whitespace, deterministic replay and reduced-budget
  end-to-end checks.

## Decisions That Must Be Frozen

The following decisions may be calibrated during Phase 2, but none may remain
implicit when reported experiment runs start:

1. The squashed or otherwise bounded Gaussian distribution, numerical epsilon,
   standard-deviation parameterization and exact log-probability.
2. Network activation, initialization and fixed critic architecture.
3. Discount and GAE factors, return/advantage normalization, entropy term,
   gradient clipping and bootstrap treatment for each episode ending.
4. Optimizers, learning rates, rollout or trajectory batch sizes, minibatch
   sizes, epochs and update frequency for each algorithm.
5. Training interaction budget, evaluation cadence, checkpoint cadence and
   logging cadence.
6. The fixed Experiment 1 track and its geometry-only selection procedure.
7. Pre-experiment configuration, reported experiment and evaluation seed
   identities.
8. The scripted reference controller and the convergence threshold derived
   without consulting reported experiment outcomes.
9. The exact PPO actor-size selection rule applied after Experiment 1.
10. Training, validation and test track-pool sizes and deterministic seed
    namespaces for Experiment 2.
11. The hardware, execution device, environment worker count, thread settings
    and timing boundaries used for computational-cost comparisons.
12. The quantitative trigger for adding lateral grip and, if triggered, the
    mathematical grip model and every new physical constant.
13. Result paths, run identifiers, schemas, retained checkpoints and trajectory
    sampling cadence.

`docs/EXPERIMENT.md` distinguishes decisions already approved from decisions to
be locked by the preparatory configuration work. Changing a frozen decision
requires a dated amendment applied uniformly to every affected reported cell.

## Execution Rules

Each numbered step below is a separate task and commit. Before implementing a
step:

1. restate its scope and list the files to be touched;
2. resolve every decision marked pending for that step;
3. obtain confirmation;
4. implement the smallest complete increment;
5. run the focused validation gate and all existing tests;
6. update `docs/DIARY.md`; and
7. commit to `main` using the project commit convention.

All RNG-dependent entry points accept `--seed`. Training and pre-experiment
commands also record the seed-derived stream identities they actually use. A
failed validation gate is evidence about the current layer and must be diagnosed
before advancing to a more complex algorithm.

## Practical Execution Steps

### 0. Specify the Learning Contract and Freeze Protocol Fields

**Status:** Complete.

**Objective:** Translate the repository theory and experiment design into exact,
project-specific equations and an auditable decision registry.

**Work:**

- Specify the bounded continuous policy distribution, sampling transform,
  deterministic action and summed log-probability.
- Specify REINFORCE returns and objective, A2C value targets and GAE, and PPO's
  clipped objective and value update.
- Define termination and time-limit truncation bootstrapping for every target.
- Decide initialization, normalization, optimizer, loss reduction and numerical
  conventions.
- Fill the algorithm, budget, cadence, seed, fixed-track, convergence and
  multi-track split fields marked for Phase-2 locking in `docs/EXPERIMENT.md`.
- Define pre-experiment settings separately from reported experiment settings.
- Correct README guidance where it conflicts with the approved contract.

**Expected files:**

- `docs/LEARNING.md`
- `docs/EXPERIMENT.md`
- `README.md`
- `docs/DIARY.md`

**Validation gate:**

- Every implemented loss and target can be derived solely from the written
  equations.
- Every result-affecting choice has a value or an explicitly bounded
  pre-experiment selection procedure.
- Reported-experiment, pre-experiment and reduced-budget validation seed
  namespaces cannot overlap accidentally.
- The specification receives confirmation before Step 1 begins.

### 1. Add Training Configuration, Dependencies and Seed Streams

**Status:** Complete.

**Objective:** Represent all approved training choices and random streams before
models or optimizers are introduced.

**Work:**

- Add immutable serializable configurations for the actor, critic, each agent,
  rollout collection, evaluation, logging and experiment matrices.
- Make the three actor architectures named configurations while retaining the
  exact `hidden_sizes` values in serialized output.
- Add PyTorch using an explicit supported version constraint.
- Derive named NumPy, PyTorch, environment, track-schedule, minibatch and
  evaluation seeds from one root `SeedSequence`.
- Configure deterministic PyTorch behaviour for the approved device and record
  limitations that cannot be guaranteed across platforms.

**Expected files:**

- `src/configs/training.py`
- `src/configs/experiments.py`
- `src/configs/__init__.py`
- `src/utils/seeding.py`
- `src/utils/__init__.py`
- `requirements.txt`
- `tests/configs/test_training_config.py`
- `tests/utils/test_seeding.py`
- `docs/LEARNING.md`
- `docs/DIARY.md`

**Validation gate:**

- Configuration serialization is stable and contains every approved choice.
- Equal root seeds reproduce every named stream; different root seeds change
  them; deriving one stream does not consume another.
- Global NumPy and PyTorch RNG state is not modified unexpectedly.
- `pip check` passes with the pinned dependency set.

### 2. Establish Metrics, Artifact Schemas and Reference Policies

**Status:** Complete.

**Objective:** Make behaviour and resource use observable before adding a
learned policy.

**Work:**

- Define named transition, episode, update, evaluation, timing and resource
  records matching `docs/EXPERIMENT.md`.
- Define a versioned run directory containing configuration, manifest, metadata,
  JSONL or CSV metrics, checkpoints and selected evaluation trajectories.
- Record git commit and dirty status, dependency freeze, platform, processor,
  device, thread and worker information.
- Implement random-action and deterministic scripted driving references with
  explicit controller settings.
- Add deterministic reference evaluation on a generated and saved circuit.
- Ensure evaluation interactions and time are separated from training measures.

**Expected files:**

- `src/utils/metrics.py`
- `src/utils/artifacts.py`
- `src/utils/references.py`
- `src/utils/__init__.py`
- `experiments/evaluate_references.py`
- `tests/utils/test_metrics.py`
- `tests/utils/test_artifacts.py`
- `tests/utils/test_references.py`
- `tests/experiments/test_evaluate_references.py`
- `README.md`
- `docs/DIARY.md`

**Validation gate:**

- Synthetic episodes distinguish completion, crash and time-limit truncation.
- Same-seed reference evaluations produce identical actions and summaries.
- Training, evaluation and pre-experiment counters cannot be aggregated
  accidentally.
- An interrupted artifact write is either complete or detected as incomplete.
- The scripted controller makes meaningful forward progress; otherwise the
  environment/control boundary is diagnosed before learning work begins.

### 3. Implement Shared Neural Models

**Status:** Complete.

**Objective:** Add reusable actor and critic components whose capacity and
probability semantics are controlled entirely by configuration.

**Work:**

- Implement the shared MLP builder with explicit input, output, activation,
  initialization and hidden sizes.
- Implement the bounded Gaussian policy, stochastic sampling, deterministic
  evaluation and corrected vector log-probability from Step 0.
- Implement the value network through the same MLP builder.
- Report actor, critic and total trainable parameter counts independently.
- Keep racing-specific observation dimensions outside the model definitions.

**Expected files:**

- `src/models/mlp.py`
- `src/models/policies.py`
- `src/models/value.py`
- `src/models/__init__.py`
- `tests/models/test_mlp.py`
- `tests/models/test_policies.py`
- `tests/models/test_value.py`
- `docs/DIARY.md`

**Validation gate:**

- All three actor sizes construct the documented topology and parameter counts.
- Single and batched observations produce correct output shapes.
- Actions always satisfy the environment bounds and log-probabilities match
  independent numerical calculations away from transform singularities.
- Gradients reach every intended actor, dispersion and critic parameter.
- Fixed initialization and sampling seeds reproduce parameters and actions.
- Deterministic evaluation does not consume the policy sampling stream.

### 4. Implement Normalization, Rollout Records, Returns and GAE

**Status:** Complete.

**Objective:** Preserve semantic transition data and verify every target before
it is used by an optimizer.

**Work:**

- Implement running observation statistics that update only during training,
  can be saved and restored, and remain frozen during evaluation.
- Define a semantic on-policy rollout record for vector observations and
  actions, including track and episode identity.
- Compute the approved Monte Carlo returns, bootstrapped value targets and GAE.
- Preserve distinct `terminated` and `truncated` masks.
- Support complete-episode batches for REINFORCE and fixed-length rollouts
  spanning episode boundaries for A2C and PPO.

**Expected files:**

- `src/utils/normalizers.py`
- `src/utils/buffers.py`
- `src/utils/__init__.py`
- `tests/utils/test_normalizers.py`
- `tests/utils/test_buffers.py`
- `docs/DIARY.md`

**Validation gate:**

- Hand-computed trajectories match every return, target and advantage.
- Termination removes the bootstrap while truncation follows the exact approved
  convention.
- No transition is lost, duplicated or assigned to the wrong episode at a
  rollout boundary.
- Saved and restored normalizers produce identical values.
- Evaluation leaves normalization statistics unchanged.

### 5. Add the Shared On-Policy Training and Evaluation Engine

**Status:** Complete.

**Objective:** Give every algorithm the same collection, evaluation,
checkpointing and accounting boundary.

**Work:**

- Define the project-owned agent update contract for continuous vector actions.
- Implement environment collection for complete episodes and fixed rollouts.
- Implement deterministic checkpoint evaluation at exact training-step
  boundaries without contaminating training streams.
- Save and restore models, optimizers, normalizers, counters, active episode
  state where required, and RNG state.
- Measure environment collection, optimization, evaluation and total duration
  separately.
- Record environment steps as the primary budget while also retaining episodes
  and updates.

**Expected files:**

- `src/agents/types.py`
- `src/agents/__init__.py`
- `src/utils/training.py`
- `src/utils/evaluation.py`
- `src/utils/checkpointing.py`
- `src/utils/__init__.py`
- `tests/utils/test_training.py`
- `tests/utils/test_evaluation.py`
- `tests/utils/test_checkpointing.py`
- `docs/DIARY.md`

**Validation gate:**

- A fixed policy and seed reproduce transitions, counters and evaluation
  summaries.
- Resume from a checkpoint matches an uninterrupted short run on the supported
  device.
- Evaluation cadence does not change sampled training actions or final weights.
- Timing categories are non-overlapping and reconcile with recorded total time.
- Collected records contain everything required by all three agents.

### 6. Implement and Validate REINFORCE

**Status:** Pending; depends on Step 5.

**Objective:** Establish the simplest project-owned policy-gradient learner
before introducing a critic.

**Work:**

- Implement the exact Step-0 REINFORCE estimator and optimizer loss.
- Keep the implementation actor-only; do not silently introduce a learned
  critic.
- Log policy loss, entropy or dispersion diagnostics, gradient norm, weight norm
  and update magnitude.
- Add a small deterministic continuous-control validation problem.
- Connect REINFORCE to the shared training entry point.

**Expected files:**

- `src/agents/reinforce.py`
- `src/agents/__init__.py`
- `experiments/train.py`
- `tests/agents/test_reinforce.py`
- `tests/fixtures/envs/continuous_control.py`
- `tests/experiments/test_train.py`
- `docs/DIARY.md`

**Validation gate:**

- Analytical loss and gradient examples match hand calculations.
- One controlled update changes action probability in the expected direction.
- The validation problem meets its predeclared improvement gate across the
  validation seeds.
- Same-seed short runs reproduce parameters, actions and summaries.
- Failure stops progress to A2C rather than being hidden by its critic.

### 7. Implement and Validate A2C with GAE

**Status:** Pending; depends on Step 6.

**Objective:** Add synchronous actor-critic learning using the already tested
policy, value model, rollout and GAE components.

**Work:**

- Implement the approved A2C actor and critic objectives.
- Use detached GAE advantages and approved value targets without leaking critic
  gradients into the actor.
- Keep actor and critic optimizers or parameter groups exactly as specified.
- Log actor loss, critic loss, entropy, explained variance, target and advantage
  statistics, gradient norms, weight norms and update magnitude.
- Connect A2C to the same training and evaluation entry point.

**Expected files:**

- `src/agents/a2c.py`
- `src/agents/__init__.py`
- `experiments/train.py`
- `tests/agents/test_a2c.py`
- `tests/experiments/test_train.py`
- `docs/DIARY.md`

**Validation gate:**

- Analytical actor, critic and combined update cases match the approved
  equations.
- Actor and critic gradients are isolated as specified.
- `lambda=0` matches the approved one-step advantage boundary case.
- The controlled validation problem meets its predeclared improvement gate.
- Same-seed runs reproduce rollout, targets, updates and summaries.

### 8. Implement and Validate PPO

**Status:** Pending; depends on Step 7.

**Objective:** Add clipped, multi-epoch sample reuse without changing the shared
data or evaluation contracts.

**Work:**

- Store behaviour-policy log-probabilities and values at collection time.
- Implement the clipped surrogate objective, approved value objective and
  entropy term.
- Implement deterministic seeded minibatch ordering and multiple update epochs.
- Log approximate KL, clip fraction, importance-ratio statistics, entropy,
  actor and critic losses, explained variance, gradient norms, weight norms and
  update magnitude.
- Implement any approved KL stop as an explicit configuration choice.

**Expected files:**

- `src/agents/ppo.py`
- `src/agents/__init__.py`
- `experiments/train.py`
- `tests/agents/test_ppo.py`
- `tests/experiments/test_train.py`
- `docs/DIARY.md`

**Validation gate:**

- Hand-built positive and negative advantage cases verify the clipped minimum.
- A no-change policy produces unit importance ratios and zero approximate KL.
- Minibatches cover every rollout row exactly once per epoch.
- Old log-probabilities remain fixed across optimization epochs.
- The controlled validation problem meets its predeclared improvement gate.
- Same-seed short runs reproduce minibatch order and final parameters.

### 9. Add Reproducible Analysis and Reporting

**Status:** Pending; depends on Step 8.

**Objective:** Derive every table, curve and diagnostic in `docs/EXPERIMENT.md` from
machine-readable run artifacts.

**Work:**

- Aggregate at the training-seed level without treating repeated checkpoints or
  tracks as independent training runs.
- Compute final performance, learning-curve area, time and interactions to
  convergence, censoring status and between-seed dispersion.
- Compute paired algorithm, size and observation summaries where the design is
  paired.
- Plot learning curves with uncertainty, task outcomes, convergence/resource
  trade-offs and optimization diagnostics.
- Plot curvature-conditioned throttle, speed and steering without excluding
  crashes or incomplete laps.
- Make report generation deterministic and independent of file discovery order.

**Expected files:**

- `src/utils/analysis.py`
- `src/utils/plotting.py`
- `experiments/analyze_results.py`
- `tests/utils/test_analysis.py`
- `tests/utils/test_plotting.py`
- `tests/experiments/test_analyze_results.py`
- `README.md`
- `docs/DIARY.md`

**Validation gate:**

- Synthetic fixtures with known aggregates, ties, failed runs and right-censored
  convergence produce the expected summaries.
- Paired differences match run identities rather than directory ordering.
- Completed-only lap-time summaries always include completion counts.
- Re-running analysis yields byte-stable data tables and semantically identical
  plots.

### 10. Select the Fixed Circuit and Configure Learning Settings

**Status:** Pending; depends on Step 9.

**Objective:** Choose reported experiment settings without using reported roots
or held-out outcomes.

**Work:**

- Generate candidate Experiment 1 circuits and select one using only the
  predeclared geometry criteria in `docs/EXPERIMENT.md`.
- Save the selected circuit as versioned data and record its generator config
  and seed.
- Evaluate the reference policies and define the fixed convergence threshold
  from reference and pre-experiment evidence.
- Configure algorithm-specific settings with the medium actor only, equal
  interaction allowances and a dedicated pre-experiment seed namespace.
- Freeze budgets, cadences, controller settings, algorithm configurations and
  the computational-cost execution environment.
- Retain every configuration run, including failures, under explicitly named
  pre-experiment paths.

**Expected files:**

- `tracks/experiment_1.json`
- `src/configs/experiments.py`
- `docs/EXPERIMENT.md`
- `experiments/calibrate.py`
- focused orchestration tests
- `README.md`
- `docs/DIARY.md`
- small pre-experiment summaries, not reported results

**Validation gate:**

- Circuit selection can be reproduced without loading any policy result.
- Every algorithm has one frozen configuration applied to all three actor
  sizes.
- Pre-experiment and reported roots are disjoint and visibly labeled.
- The frozen convergence rule can be evaluated from logged checkpoints alone.
- No held-out Experiment 2 track participates in this step.

### 11. Diagnose Version-0 Cornering

**Status:** Pending; depends on Step 10.

**Objective:** Decide from a capable learned policy whether version-0 dynamics
make unrealistic full-speed cornering relevant to the study.

**Work:**

- Train the designated pre-experiment PPO configuration on version-0 physics.
- Require the policy to pass the capability gate before interpreting its speed
  choice.
- Replay deterministic policies and align current/preview curvature with speed,
  throttle and steering.
- Report all predeclared curvature bins, crashes and incomplete trajectories.
- Apply the quantitative grip trigger in `docs/EXPERIMENT.md` without changing it
  after seeing the traces.
- Record a decision to retain version 0 or begin the conditional grip work.

**Expected files:**

- `experiments/diagnose_cornering.py`
- focused analysis tests
- `docs/EXPERIMENT.md`
- `docs/DIARY.md`
- the approved compact pre-experiment diagnosis

**Validation gate:**

- The analyzed policy has demonstrated useful steering/progress, so high speed
  is not attributed to an untrained controller.
- The report includes unsuccessful episodes and answers whether speed or
  throttle decreases with curvature.
- The trigger calculation is reproducible from retained artifacts.
- The physics decision is recorded before an Experiment 1 reported-run manifest
  is generated.

### 12. Specify the Conditional Lateral-Grip Model

**Status:** Conditional; depends on Step 11 triggering grip work.

**Objective:** Define the smallest physically motivated change that creates a
real speed-versus-curvature constraint.

**Work:**

- Add the exact lateral-acceleration or equivalent constraint to `docs/MDP.md`.
- Define infeasible-control behaviour, physics-substep ordering, collision
  interaction and speed-bound interaction.
- Source or derive every new constant independently of desired training reward.
- Preserve an explicit selectable version-0 configuration.
- Derive straight, low-speed, boundary and high-speed curved test cases.
- Keep action, observation and reward definitions unchanged.

**Expected files:**

- `docs/MDP.md`
- `docs/EXPERIMENT.md`
- `docs/DIARY.md`

**Validation gate:**

- The model is mathematically complete and dimensionally consistent.
- Every constant and boundary case is explicit.
- Analytical examples predict unchanged straight/low-speed motion and the
  intended high-speed limitation.
- The specification receives confirmation before implementation.

If Step 11 does not trigger grip work, record this step as not applicable and
continue with version-0 physics; do not invent a dynamics change merely because
it appeared in the roadmap.

### 13. Implement and Revalidate Conditional Grip

**Status:** Conditional; depends on Step 12.

**Objective:** Implement only the approved constraint and freeze the physics
version used by both experiments.

**Work:**

- Extend vehicle configuration and transition kernel with the approved rule.
- Keep version-0 replay selectable and deterministic.
- Surface physics version and grip activation diagnostics in environment info
  and run metadata.
- Repeat focused analytical tests and the version-0 cornering diagnosis with the
  same reward, observation and pre-experiment protocol.
- Confirm useful progress and a meaningful speed-versus-curvature response
  before freezing the reported experiment environment.

**Expected files:**

- `src/configs/environment.py`
- `src/envs/vehicle/kernel.py`
- `src/envs/racing/environment.py`
- `tests/configs/test_environment_config.py`
- `tests/envs/test_dynamics.py`
- `tests/envs/test_racing_env.py`
- `experiments/diagnose_cornering.py`
- `README.md`
- `docs/DIARY.md`

**Validation gate:**

- Version-0 transition and replay tests still pass when version 0 is selected.
- Straight and approved low-speed behaviour is unchanged.
- High-speed curve cases obey the exact specified constraint.
- Gymnasium conformance and the complete existing suite pass.
- The repeated pre-experiment diagnosis meets the capability gate and reports
  the intended physical effect without reward changes.

### 14. Assemble and Validate Experiment 1 End to End

**Status:** Pending; depends on Step 11 or, when triggered, Step 13.

**Objective:** Prove that the frozen single-circuit 3-by-3 design can be launched
and analyzed without manual intervention.

**Work:**

- Build the nine-cell manifest from three algorithms and three actor sizes.
- Expand the manifest over the five paired reported roots without launching the
  full reported-run budget.
- Validate that critic size, environment, track, budget and evaluation protocol
  remain invariant where required.
- Run a reduced-budget end-to-end validation job for every unique
  algorithm/size cell under a dedicated validation root.
- Aggregate the validation artifacts and generate every Experiment 1 table/plot
  with explicit reduced-budget watermarks.
- Add commands for launching individual cells, the complete matrix and analysis.

**Expected files:**

- `src/configs/experiments.py`
- `experiments/run_experiment.py`
- `experiments/analyze_results.py`
- `tests/experiments/test_experiment_1.py`
- `README.md`
- `docs/DIARY.md`
- compact reduced-budget validation artifacts only

**Validation gate:**

- The manifest contains exactly 45 unique reported runs: nine cells by five
  roots.
- Actor size changes actor parameters but not critic architecture or unrelated
  configuration.
- Every reduced-budget cell trains, evaluates, checkpoints and reloads
  successfully.
- Analysis includes failed/non-converged cells instead of silently dropping
  them.
- No reduced-budget validation artifact can be loaded as reported experiment
  data without an explicit schema error.

### 15. Implement LiDAR as an Interchangeable Observation

**Status:** Pending; depends on Step 14.

**Objective:** Add the second observation representation while preserving one
shared dynamics and lifecycle core.

**Work:**

- Implement the 16-ray, 200-degree, normalized LiDAR observation specified in
  `docs/MDP.md` and `docs/TRACK.md`.
- Query all potentially intersecting boundary segments and return the nearest
  valid hit or maximum range.
- Expose `observation="frenet"|"lidar"` through an environment construction
  boundary or Gymnasium observation wrapper.
- Include speed and ranges in the exact specified order and dtype.
- Keep reward, action, transition, collision, finish and progress logic shared.

**Expected files:**

- `src/configs/environment.py`
- `src/envs/observations/lidar.py`
- `src/envs/observations/__init__.py`
- `src/envs/racing/environment.py` or a dedicated observation wrapper
- `tests/envs/test_lidar.py`
- `tests/envs/test_racing_env.py`
- `README.md`
- `docs/DIARY.md`

**Validation gate:**

- Analytical straight-wall, angled-wall, no-hit, maximum-range and nearest-hit
  cases match geometry calculations.
- Observation shape, ordering, normalization, dtype and declared space agree.
- Same state and action sequence produces identical physical outcomes under
  Frenet and LiDAR observation modes.
- Both modes pass Gymnasium conformance and deterministic reset/replay.

### 16. Add Deterministic Multi-Track Training and Held-Out Splits

**Status:** Pending; depends on Step 15.

**Objective:** Train from procedural circuit variation while protecting fixed
validation and test circuits from leakage.

**Work:**

- Define disjoint seed namespaces for development, training, validation and
  test tracks using the frozen generator configuration.
- Select each training episode's circuit from a deterministic schedule derived
  independently from policy and minibatch RNG.
- Generate tracks lazily and cache prepared geometry without changing the
  logical schedule.
- Save enough scheduler and active-track state for exact checkpoint resume.
- Evaluate on fixed validation/test seed lists without updating normalization.
- Record generator config, track seed and split for every episode and evaluation.

**Expected files:**

- `src/envs/racing/factory.py`
- `src/configs/experiments.py`
- `src/utils/training.py`
- `src/utils/evaluation.py`
- `tests/envs/test_environment_factory.py`
- `tests/utils/test_multitrack_training.py`
- `docs/EXPERIMENT.md`
- `docs/DIARY.md`

**Validation gate:**

- Split seed sets are disjoint and reproducible.
- Frenet/LiDAR pairs receive identical training-track schedules for a root seed.
- Resume produces the same subsequent track sequence and training transitions.
- Evaluation never mutates the training schedule or normalizer.
- Every final test result identifies its track and no test track appears in a
  training or calibration artifact.

### 17. Assemble and Validate Experiment 2 End to End

**Status:** Pending; depends on Step 16.

**Objective:** Prove the paired PPO Frenet-versus-LiDAR generalization study can
run and be analyzed end to end.

**Work:**

- Apply the predeclared PPO actor-size selection rule to Experiment 1-shaped
  fixture results, then verify it against actual results when they exist.
- Build paired Frenet and LiDAR manifests with the same five roots, training
  schedules, budget, PPO configuration and held-out circuits.
- Run reduced-budget end-to-end validation training on multiple generated
  circuits in each observation mode.
- Evaluate both validation policies on disjoint held-out tracks.
- Generate paired task, learning, resource and generalization summaries with
  reduced-budget watermarks.

**Expected files:**

- `src/configs/experiments.py`
- `experiments/run_experiment.py`
- `experiments/analyze_results.py`
- `tests/experiments/test_experiment_2.py`
- `README.md`
- `docs/DIARY.md`
- compact reduced-budget validation artifacts only

**Validation gate:**

- The reported-run manifest contains exactly two observation conditions for
  every root and no unintended configuration difference.
- Track schedules and held-out evaluations pair correctly across observations.
- Both reduced-budget runs collect, update, checkpoint, resume and evaluate.
- Analysis reports seed-level aggregates, track-level distributions and
  train-to-test generalization gaps.
- Test-track outcomes cannot influence network selection or normalization.

### 18. Run Phase-2 Acceptance and Freeze the Reported-Run Manifest

**Status:** Pending; depends on Step 17.

**Objective:** Demonstrate that the repository is ready for unattended,
comparable reported experiment runs.

**Work:**

- Run dependency, formatting, linting, type, compilation, test and whitespace
  checks.
- Run deterministic same-seed training/resume/evaluation checks for all agents.
- Execute reduced Experiment 1 and Experiment 2 matrices from a fresh result
  directory.
- Regenerate all tables and plots solely from the new raw artifacts.
- Verify frozen manifests, dependency freeze, hardware/software metadata and
  separation between pre-experiment and reported results.
- Update README commands and diary results.
- Mark the experiment protocol frozen with a dated revision and checksum.

**Expected files:**

- `experiments/phase2_acceptance.py`
- `tests/experiments/test_phase2_acceptance.py`
- `docs/EXPERIMENT.md`
- `README.md`
- `docs/DIARY.md`
- corrections only if acceptance exposes a defect

**Validation gate:**

```bash
./.venv/bin/python -m pip check
./.venv/bin/python -m pytest
./.venv/bin/python experiments/phase2_acceptance.py --seed 0
```

- All automated and static checks pass.
- Same-seed reduced-budget training and evaluation reproduce on the supported
  machine.
- All 45 Experiment 1 run specifications and all paired Experiment 2 run
  specifications can be enumerated before execution.
- Required metrics and plots are produced even when fixture runs crash or never
  converge.
- No required decision remains only in local shell history, console output or an
  uncommitted file.

## After Phase 2

The next work is execution, not another implementation phase:

1. run the frozen Experiment 1 matrix;
2. analyze and report policy-size effects and the secondary algorithm
   comparison;
3. apply the frozen PPO size-selection rule;
4. run the paired Experiment 2 matrix;
5. analyze observation and circuit-generalization effects; and
6. write the final scientific report, retaining negative and inconclusive
   findings.

Any later addition such as recurrent LiDAR policies, randomized starts, a new
reward, additional dynamics or another algorithm requires a new plan and must
not be inserted into the frozen comparison retroactively.
