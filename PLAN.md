# Project Plan — Phase 2: Learned Baseline and Grip Validation

## Goal

Build the first reproducible, project-owned learning path for the racing
environment and use it to test the most important known limitation of the
version-0 dynamics. Phase 2 must:

- establish deterministic non-learning reference policies and task metrics;
- implement and validate a configurable Gaussian neural policy;
- implement vanilla REINFORCE from the equations approved for this project;
- demonstrate a measurable learning signal on the unchanged Phase-1
  environment;
- observe and quantify how the learned policy behaves around curves before
  changing the physics;
- add a lateral-grip constraint only after the version-0 behaviour has been
  demonstrated;
- repeat the learning comparison without changing the reward or unrelated
  training choices; and
- measure the effect of adding a learned value baseline only after vanilla
  REINFORCE has its own recorded result.

This phase is a controlled model-validation study, not the final network-size
experiment. Its purpose is to build confidence that later algorithm and policy
complexity comparisons measure learning rather than an untested environment,
reward or training pipeline.

## Additive Development Rule

Every advancement in this roadmap depends on evidence from the simpler system
immediately before it. A passing unit test is necessary but is not enough when a
step makes a behavioural claim: the relevant training or evaluation result must
also be recorded.

In particular:

1. Do not add a learned value baseline until vanilla REINFORCE has been tested
   and measured.
2. Do not add lateral grip until a learned policy on version-0 physics has been
   evaluated and its speed and throttle around curves have been inspected.
3. Do not tune the reward to encourage braking while version-0 physics still
   makes full-throttle cornering feasible.
4. Do not add a more advanced algorithm to hide a failing policy, rollout,
   return or seeding implementation.
5. Do not proceed through a failed validation gate. Diagnose the current layer,
   amend the relevant specification if necessary, and obtain confirmation
   before continuing.

The central comparison is therefore:

```text
tested environment
  -> tested reference policies and metrics
  -> tested policy primitives
  -> tested vanilla REINFORCE
  -> observed version-0 learned behaviour
  -> specified and tested grip constraint
  -> repeated learning comparison
  -> tested variance-reduction baseline
```

## Authoritative Specifications

Implementation decisions must agree with:

- [`docs/MDP.md`](docs/MDP.md) for the environment state, observation, action,
  version-0 dynamics, reward and episode lifecycle;
- [`docs/TRACK.md`](docs/TRACK.md) for track geometry, Frenet projection,
  progress and lap completion;
- the learning specification introduced in Step 0 for the policy distribution,
  objective, returns, optimization, evaluation and reproducibility contract;
- [`AGENTS.md`](AGENTS.md) for workflow, testing, determinism, dependency and
  documentation rules; and
- the archived Phase-1 roadmap in
  [`docs/old-plans/phase-1-racing-environment-mvp.md`](docs/old-plans/phase-1-racing-environment-mvp.md)
  for the accepted environment baseline.

Algorithm code must be implemented from the approved project equations. The
README's current algorithm discussion is useful motivation but is not a complete
algorithm specification.

## Scope Boundaries

Phase 2 includes only the minimum learning and physics changes needed for the
controlled comparison above. It deliberately excludes:

- A2C, GAE and PPO;
- the formal policy-network size sweep;
- finite vehicle footprint, tire slip, aerodynamic drag and steering-rate
  limits;
- LiDAR observations;
- randomized starts and curricula;
- vectorized training environments;
- multi-track training and held-out-track generalization; and
- reward tuning not justified by the gated comparison.

These remain candidates for later plans. Observation normalization, entropy
regularization, return scaling and other common training additions are also not
enabled silently. Each must be introduced as a separately measured increment if
the simpler training path demonstrates a concrete need.

## Definition of Done

Phase 2 is complete only when all of the following are true:

- One command can evaluate deterministic reference policies and emit stable
  episode summaries for a fixed seed and configuration.
- Policy, rollout and training random-number streams are independently derived
  from one explicit root seed and reproduce the same run on the same supported
  software and hardware configuration.
- The shared MLP and bounded continuous-action policy pass focused numerical,
  shape, gradient and determinism tests.
- Vanilla REINFORCE passes small analytical tests and demonstrates learning on a
  deliberately simple continuous-control problem before racing training begins.
- A version-0 racing run shows a predeclared improvement over the non-learning
  reference selected in the experiment specification.
- Deterministic evaluations record return, normalized lap progress, completion,
  crash, lap time, action, speed and curvature diagnostics.
- The version-0 report explicitly establishes whether the learned policy slows
  for curves. Lateral-grip work begins only after this evidence exists.
- The grip model and every new physical constant are approved in `docs/MDP.md`
  before implementation.
- Low-speed behaviour remains compatible with version 0, while focused tests
  demonstrate the intended high-speed grip limitation.
- The before/after grip comparison holds reward, observation, track, seeds,
  policy architecture, optimizer and training budget fixed unless a documented
  incompatibility makes that impossible.
- The grip-limited learned policy is evaluated with the same metrics. If braking
  does not emerge, the result is treated as a failed behavioural gate and is
  diagnosed before reward changes or further algorithms are attempted.
- The learned value baseline is compared with vanilla REINFORCE using the same
  environment version and evaluation protocol.
- All experiment configurations and summaries needed to reproduce the claims
  are retained in a documented, machine-readable form; large transient
  checkpoints are not committed by default.
- The complete automated and static validation suite passes, and the README and
  diary explain how to reproduce the phase acceptance run.

## Decisions That Must Be Resolved Explicitly

No implementation step may silently choose the following values or semantics:

1. The bounded Gaussian action construction and its exact log-probability.
2. Whether returns use the provisional `gamma=0.9995`, `gamma=1`, or an approved
   comparison before fixing one value.
3. Policy initialization, activation, initial log standard deviation, optimizer
   and learning rate.
4. Episode or transition batching, training budget and update frequency.
5. The fixed training track, evaluation episodes, root seeds, reference policy,
   success threshold and rules for declaring a learning improvement.
6. The result directory, run identifier and machine-readable log schema.
7. The mathematical lateral-grip rule and every associated physical constant.
8. The criterion used to compare vanilla REINFORCE with the learned value
   baseline.

Step 0 resolves items 1–6. The version-0 evidence step resolves the measurement
needed before item 7 can be proposed. The value-baseline step resolves item 8
before implementing that advancement.

## Execution Rules

Each numbered step below is a separate task and commit. Before implementing a
step:

1. restate its scope and list the files to be touched;
2. resolve every decision marked pending for that step;
3. obtain confirmation;
4. implement the smallest complete increment;
5. run both the focused validation gate and all existing tests;
6. update `docs/DIARY.md`; and
7. commit to `main` using the project commit convention.

Training runs used as validation gates must identify their configuration, root
seed, environment steps and wall-clock duration. A failed or inconclusive run is
evidence to investigate the current layer; it is not permission to skip ahead.

## Practical Execution Steps

### 0. Specify the Learning and Evaluation Contract

**Status:** Pending.

**Objective:** Turn the high-level algorithm notes into an authoritative,
reviewable specification before adding learning code.

**Work:**

- Write the exact bounded policy distribution, sampling transform and
  log-probability equations.
- Specify trajectory return-to-go calculation, terminal versus truncated
  handling and the selected discount protocol.
- Specify the vanilla REINFORCE objective and sign convention used by the
  optimizer.
- Define policy initialization, optimizer, learning rate, batching, update
  frequency and training budget as explicit experiment configuration.
- Define the root-seed derivation for environment, action sampling, parameter
  initialization and evaluation.
- Select the fixed generated track and reference-policy seeds used during this
  phase.
- Define deterministic evaluation, the reference comparison, the learning gate
  and the metrics required for behavioural analysis.
- Define the run directory and machine-readable configuration, episode-summary,
  update-summary and trajectory-sample schemas.
- Update the README only where its current informal recommendations conflict
  with the approved specification.

**Expected files:**

- `docs/LEARNING.md`
- `README.md` if clarification is required
- `docs/DIARY.md`

**Validation gate:**

- Every equation needed to implement vanilla REINFORCE is present and internally
  consistent.
- Every core training hyperparameter has an approved value or an explicitly
  approved finite comparison; no placeholder enters source code.
- Truncation bootstrapping semantics are unambiguous for vanilla Monte Carlo
  returns.
- The learning and behavioural gates can be calculated solely from the defined
  log records.
- The specification receives confirmation before Step 1 begins.

### 1. Add Deterministic Episode Metrics and Reference Policies

**Status:** Pending; depends on Step 0.

**Objective:** Establish observable, reproducible environment behaviour before
introducing a neural policy.

**Work:**

- Define named episode and trajectory diagnostic records.
- Record return, episode length, elapsed time, normalized progress, completion,
  crash, lap time, actions, speed and preview curvature.
- Implement the approved non-learning references, including a random-action
  policy and simple deterministic controls appropriate to the approved
  comparison.
- Add a thin evaluation experiment that runs the selected fixed track and writes
  the approved summaries.
- Keep reference-policy parameters explicit; do not bury controller gains or
  action values in implementation code.

**Expected files:**

- `src/utils/metrics.py`
- `src/utils/seeding.py`
- `src/utils/__init__.py`
- `experiments/evaluate_baselines.py`
- `tests/utils/test_metrics.py`
- `tests/utils/test_seeding.py`
- `tests/experiments/test_evaluate_baselines.py`
- `README.md`
- `docs/DIARY.md`

**Validation gate:**

- Repeated runs with the same root seed produce identical reference-policy
  actions and episode summaries.
- Synthetic episode tests distinguish crash, completion and truncation metrics.
- The fixed-track reference results are saved and inspectable.
- At least one deterministic control sanity check establishes that observations,
  actions and progress move in the expected directions. If no approved
  reference can make meaningful progress, diagnose the environment/control
  interface before adding learning.

### 2. Add Explicit Training Configuration and Seed Streams

**Status:** Pending; depends on Step 1.

**Objective:** Represent every approved training choice and random stream before
implementing models or updates.

**Work:**

- Add immutable configuration records for the policy, vanilla REINFORCE run and
  deterministic evaluation.
- Serialize configurations in a stable form suitable for result metadata.
- Derive independent NumPy, environment and PyTorch seeds from one root
  `SeedSequence` without changing global random state unexpectedly.
- Add PyTorch to the dependency manifest using the approved explicit version
  constraint.
- Configure deterministic PyTorch behaviour to the extent supported by the
  selected execution device and document any remaining platform limitation.

**Expected files:**

- `src/configs/training.py`
- `src/configs/__init__.py`
- `src/utils/seeding.py`
- `requirements.txt`
- `tests/configs/test_training_config.py`
- `tests/utils/test_seeding.py`
- `docs/LEARNING.md`
- `docs/DIARY.md`

**Validation gate:**

- Configuration serialization is stable and includes every approved choice.
- Equal root seeds derive equal independent streams; changing the root seed
  changes them.
- Model initialization and sampled action sequences reproduce for a fixed seed.
- `pip check` passes with the updated dependency manifest.

### 3. Implement and Test the Shared MLP

**Status:** Pending; depends on Step 2.

**Objective:** Introduce the single network-size-controlled neural primitive
without coupling it to a learning algorithm.

**Work:**

- Implement `make_mlp` from the approved activation and initialization
  specification.
- Make `hidden_sizes` the only structural depth/width input.
- Keep input/output dimensions explicit and avoid racing-specific assumptions in
  the model builder.
- Test parameter shapes, forward shapes, gradients, initialization and seeded
  reproducibility.

**Expected files:**

- `src/models/mlp.py`
- `src/models/__init__.py`
- `tests/models/test_mlp.py`
- `docs/DIARY.md`

**Validation gate:**

- Empty, shallow and multi-layer approved configurations build the documented
  topology.
- Batched and single-observation forward passes have the expected shapes.
- Backpropagation reaches every trainable parameter.
- Fixed initialization seeds reproduce parameters exactly on the supported
  device.

### 4. Implement and Test the Bounded Gaussian Policy

**Status:** Pending; depends on Step 3.

**Objective:** Map Frenet observations to valid continuous racing actions with
the exact probability semantics approved in Step 0.

**Work:**

- Implement the policy mean using the shared MLP.
- Implement the approved standard-deviation parameterization.
- Implement stochastic sampling, deterministic mean-action evaluation and
  log-probability calculation for bounded vector actions.
- Sum log-probabilities over the action dimensions exactly once.
- Keep observation normalization absent unless separately justified and
  approved.

**Expected files:**

- `src/models/policies.py`
- `src/models/__init__.py`
- `tests/models/test_policies.py`
- `docs/DIARY.md`

**Validation gate:**

- Sampled and deterministic actions always satisfy the environment action
  bounds and dtype/shape contract.
- Numerical distribution tests agree with the approved transform and Jacobian
  equations away from singular boundaries.
- Deterministic evaluation does not consume the sampling random stream.
- Gradients reach both mean-network and approved dispersion parameters.
- Seeded samples and log-probabilities reproduce.

### 5. Implement Rollout Collection and Monte Carlo Returns

**Status:** Pending; depends on Step 4.

**Objective:** Test data collection and return computation independently of the
optimizer.

**Work:**

- Define semantic per-transition and complete-episode records.
- Collect observations, sampled actions, rewards, termination and truncation
  without losing their meanings in anonymous tuples.
- Compute return-to-go using the exact Step-0 equations.
- Preserve Gymnasium termination and truncation semantics.
- Produce the episode metrics defined in Step 1 from the same rollout.

**Expected files:**

- `src/utils/buffers.py`
- `src/utils/training.py`
- `src/utils/__init__.py`
- `tests/utils/test_buffers.py`
- `tests/utils/test_training.py`
- `docs/DIARY.md`

**Validation gate:**

- Hand-computed finite trajectories match every calculated return.
- Terminated and time-truncated examples follow the approved semantics.
- No transition is dropped or duplicated at episode boundaries.
- A fixed policy and seed reproduce the complete rollout and summary.

### 6. Implement Vanilla REINFORCE and Verify the Base Learner

**Status:** Pending; depends on Step 5.

**Objective:** Prove the simplest approved policy-gradient implementation before
using it to interpret racing behaviour.

**Work:**

- Implement the project-owned vanilla REINFORCE loss and update.
- Do not add a critic, learned baseline, GAE, PPO clipping, entropy bonus or
  observation normalization.
- Add a deliberately small continuous-control test environment or analytical
  objective whose improvement can be established cheaply and deterministically.
- Log update loss, return statistics, gradient diagnostics and environment-step
  count using the Step-0 schema.
- Add a thin training entry point driven entirely by explicit configuration and
  root seed.

**Expected files:**

- `src/agents/reinforce.py`
- `src/agents/__init__.py`
- `src/utils/training.py`
- `experiments/train_reinforce.py`
- `tests/agents/test_reinforce.py`
- `tests/fixtures/envs/` if a test environment is needed
- `tests/experiments/test_train_reinforce.py`
- `README.md`
- `docs/DIARY.md`

**Validation gate:**

- Analytical loss and gradient cases match hand calculations.
- One update changes parameters in the expected direction.
- Across the predeclared fixed seeds, the simple continuous-control task meets
  its predeclared improvement threshold.
- Repeating a same-seed short run reproduces parameters, summaries and sampled
  actions on the supported device.
- If the learner fails this gate, racing training does not begin.

### 7. Establish the Version-0 Learned Racing Baseline

**Status:** Pending; depends on Step 6.

**Objective:** Demonstrate and measure policy learning on the unchanged Phase-1
environment before altering its physics.

**Work:**

- Train vanilla REINFORCE on the approved fixed track, configuration and seed
  set.
- Evaluate checkpoints using deterministic actions and the fixed evaluation
  protocol.
- Compare with the approved non-learning reference using return, normalized
  progress, completion and crash rate rather than training loss alone.
- Retain the exact configuration, summaries and selected evaluation trajectories
  required by the learning gate.
- Do not change the reward, observation, initial-state distribution or dynamics
  during this step.

**Expected files:**

- `experiments/train_reinforce.py`
- `experiments/evaluate_policy.py`
- focused tests for any new orchestration behaviour
- `README.md`
- `docs/DIARY.md`
- the approved small Phase-2 result summaries

**Validation gate:**

- The predeclared learning criterion is met across the approved seeds.
- Deterministic evaluation outperforms the selected non-learning reference on
  the approved primary task metric.
- Complete configuration and environment-step counts accompany the result.
- If vanilla REINFORCE shows no reliable racing learning signal, stop and
  diagnose this layer. Do not add grip. Any proposal to introduce variance
  reduction earlier than planned requires a documented result and separate
  approval.

### 8. Diagnose Learned Speed Choice Around Curves

**Status:** Pending; depends on Step 7.

**Objective:** Establish empirical evidence for or against the documented
version-0 full-throttle limitation.

**Work:**

- Replay the selected learned policy deterministically without updating it.
- Align throttle, steering and speed with current and preview curvature.
- Report speed and throttle by predeclared absolute-curvature bins or another
  approved equivalent statistic.
- Plot or tabulate representative trajectories and action/speed traces.
- Distinguish a policy that has not learned useful steering from a capable
  policy that deliberately keeps accelerating through curves.
- Record crashes and incomplete laps so survivorship does not bias the
  conclusion.

**Expected files:**

- `src/utils/plotting.py`
- `experiments/analyze_policy.py`
- `tests/utils/test_plotting.py`
- `tests/experiments/test_analyze_policy.py`
- `README.md`
- `docs/DIARY.md`
- the approved small Phase-2 analysis outputs

**Validation gate:**

- The analysis is reproducible from a saved configuration and policy checkpoint.
- The report explicitly answers whether the learned policy reduces throttle or
  speed as curvature increases.
- The policy has already met Step 7's learning criterion, so the result is not an
  artifact of an untrained controller.
- Grip specification may begin only if the evidence confirms that version-0
  dynamics permit unrealistic cornering behaviour relevant to the objective. If
  it does not, stop and revise the physical hypothesis before changing dynamics.

### 9. Specify the Minimum Lateral-Grip Model

**Status:** Pending; depends on Step 8 confirming the model limitation.

**Objective:** Define the smallest physical change that creates a real
speed-versus-curvature trade-off without introducing a full dynamic tire model.

**Work:**

- Add the proposed lateral-acceleration or equivalent grip equation to
  `docs/MDP.md`.
- State how infeasible steering behaves, how the rule is evaluated at physics
  substeps and how it interacts with collision and speed bounds.
- Define every new physical constant with source or derivation; do not select a
  hidden threshold from desired reward behaviour.
- Specify whether version-0 dynamics remain selectable for the controlled
  comparison and how the environment version is logged.
- Derive low-speed compatibility, straight-line and high-speed curved-motion
  test cases before implementation.
- Keep the reward and observation unchanged.

**Expected files:**

- `docs/MDP.md`
- `PLAN.md` if implementation details need clarification
- `docs/DIARY.md`

**Validation gate:**

- The new model is mathematically complete and dimensionally consistent.
- All constants and boundary cases are explicit.
- The model predicts unchanged straight/low-speed cases and constrained
  high-speed cornering in the specified analytical examples.
- The specification receives confirmation before Step 10 begins.

### 10. Implement and Test the Grip-Limited Transition

**Status:** Pending; depends on Step 9.

**Objective:** Add only the approved grip behaviour while preserving all
unrelated Phase-1 semantics.

**Work:**

- Extend vehicle configuration and the kinematic transition with the approved
  grip rule.
- Preserve an explicit version-0 configuration for the before/after comparison
  if required by the specification.
- Surface the environment physics version and useful grip diagnostics in run
  metadata or `info` as approved.
- Update the MDP and README descriptions if implementation exposes a confirmed
  clarification.

**Expected files:**

- `src/configs/environment.py`
- `src/envs/vehicle/kernel.py`
- `src/envs/racing/environment.py` if diagnostics are exposed there
- `tests/configs/test_environment_config.py`
- `tests/envs/test_dynamics.py`
- `tests/envs/test_racing_env.py`
- `README.md`
- `docs/DIARY.md`

**Validation gate:**

- Existing version-0 transition tests still pass in the version-0 configuration.
- Straight-line and approved low-speed results remain unchanged.
- Analytical high-speed curved cases exhibit the exact specified limitation.
- Same-seed same-version action replays remain deterministic.
- Gymnasium conformance and the complete Phase-1 acceptance suite pass.

### 11. Repeat the Learned Comparison with Grip

**Status:** Pending; depends on Step 10.

**Objective:** Isolate the behavioural effect of the physics change using the
same learner and reward.

**Work:**

- Train from scratch using the approved version-0 configuration with only the
  physics version changed to the grip-limited model.
- Hold track, seed set, policy architecture, optimizer, reward, observation,
  training budget and evaluation protocol fixed.
- Report any forced deviation before running rather than after seeing results.
- Repeat deterministic performance and curvature-conditioned behaviour
  analysis.
- Compare completion, crash rate, lap time, progress, throttle and speed against
  the recorded version-0 baseline.

**Expected files:**

- `experiments/train_reinforce.py`
- `experiments/evaluate_policy.py`
- `experiments/analyze_policy.py`
- focused tests for comparison orchestration
- `README.md`
- `docs/DIARY.md`
- the approved small before/after result summaries

**Validation gate:**

- The comparison metadata proves that the physics configuration is the only
  intended experimental change.
- The grip-limited policy meets the predeclared learning criterion.
- Evaluation establishes whether speed or throttle now decreases with relevant
  curvature while retaining useful progress or lap completion.
- If braking does not emerge, stop and diagnose dynamics, observation and
  learning evidence. Do not compensate by changing the reward or adding PPO.

### 12. Add and Measure a Learned Value Baseline

**Status:** Pending; depends on Step 11.

**Objective:** Add the first variance-reduction mechanism only after vanilla
REINFORCE and the environment comparison are understood.

**Work:**

- Specify the value target, loss, optimizer relationship, update count and
  advantage construction in `docs/LEARNING.md`.
- Implement a value network using the shared MLP builder.
- Preserve vanilla REINFORCE as a selectable, tested configuration.
- Compare vanilla and baseline variants on one approved physics version with
  paired seeds and the same training budget.
- Report return variance and the predeclared sample-efficiency or reliability
  measure; do not claim improvement from a single best seed.

**Expected files:**

- `docs/LEARNING.md`
- `src/models/value.py`
- `src/agents/reinforce.py`
- `src/configs/training.py`
- `tests/models/test_value.py`
- `tests/agents/test_reinforce.py`
- `experiments/train_reinforce.py`
- `README.md`
- `docs/DIARY.md`
- the approved small algorithm-comparison summaries

**Validation gate:**

- Hand-computed value targets, value loss and advantages match the approved
  equations.
- Disabling the baseline follows the already tested vanilla path.
- The paired-seed comparison reports the approved variance and efficiency
  criteria whether or not the baseline improves them.
- A negative result is retained as evidence rather than hidden by advancing to
  another algorithm.

### 13. Run the Phase-2 Acceptance Pass

**Status:** Pending; depends on Step 12.

**Objective:** Prove that the learning baseline, behavioural diagnosis and grip
comparison are reproducible enough to support a later algorithm or network-size
study.

**Work:**

- Run dependency, formatting, linting, type, compilation, test and whitespace
  checks.
- Run deterministic short training/replay checks suitable for routine
  acceptance.
- Verify the committed experiment summaries against their saved configurations.
- Reproduce the version-0 behaviour diagnosis and grip-limited comparison using
  the documented commands and retained artifacts.
- Record dependency versions, hardware/software context, environment-step counts
  and wall-clock durations.
- Update the README and diary with commands, outcomes and remaining limitations.

**Expected files:**

- `experiments/phase2_acceptance.py`
- `tests/experiments/test_phase2_acceptance.py`
- `README.md`
- `docs/DIARY.md`
- corrections only if acceptance exposes a defect

**Validation gate:**

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest
```

- All automated and static checks pass.
- Same-seed smoke training and evaluation reproduce on the supported machine.
- The retained results support every Definition-of-Done claim without relying
  on an undocumented local file.
- The phase records both positive and negative experimental results honestly.

## Deferred Work

After Phase 2, create a new plan rather than extending this roadmap silently.
Candidate later phases are:

1. A2C with GAE, followed by PPO as separate measured algorithm increments;
2. the formal network-depth/width experiment after fixing all other choices;
3. finite vehicle footprint and further vehicle-dynamics refinements;
4. observation normalization or other training improvements justified by
   recorded evidence;
5. LiDAR observations;
6. randomized starts and curricula; and
7. multi-track training and held-out-track generalization.
