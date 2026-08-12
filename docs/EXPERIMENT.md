# Experimental Protocol

## Purpose and status

This document specifies two reported experiments:

1. the effect of actor-network size on one fixed circuit; and
2. PPO generalization across circuits with Frenet versus LiDAR observations.

Each experiment is presented in full, including its conditions, random roots,
training procedure, recorded data, outcomes and analysis. Repeated details are
intentional: understanding one experiment should not require jumping between
several common-protocol sections.

Work performed before these experiments may select usable learning rates,
validate the software path and choose the fixed circuit. Those runs are
development evidence, not observations in either reported experiment, and their
results are never pooled with the reported results.

**Protocol revision:** 2026-08-12 grip-limited physics contract.

## Before either experiment

This section contains configuration and validation rules that must be completed
before the reported experiment manifests are frozen. They are guardrails for
obtaining executable settings, not a third scientific experiment.

### Learning configuration check

The exact policy, model, target and loss definitions are in
[`LEARNING.md`](LEARNING.md). The actor-size factor is already fixed, but the
course equations do not determine learning rates. Short pre-experiment runs
therefore compare only the following finite candidates with the medium
`(64, 64)` actor:

- REINFORCE actor rate:
  $\{10^{-4},3\cdot10^{-4},10^{-3}\}$;
- A2C actor/critic rates:
  $\{(10^{-4},3\cdot10^{-4}),(3\cdot10^{-4},10^{-3})\}$;
- PPO actor/critic rates:
  $\{(10^{-4},3\cdot10^{-4}),(3\cdot10^{-4},10^{-3})\}$.

These candidates are engineering scales around Adam's $10^{-3}$ suggested
default, reduced where the noisy policy-gradient objective benefits from
smaller steps. They do not come from the policy-gradient theorem.

Each candidate receives 250,000 interactions on each of three dedicated roots.
This allowance is long enough to reveal immediate divergence and early progress
without pretending to establish final performance. Every candidate consumes the
same allowance and none stops early. Select lexicographically by:

1. number of final deterministic policies that complete the lap;
2. mean final maximum normalized progress;
3. mean final deterministic return; and
4. smaller actor rate, then smaller critic rate.

This is a pragmatic rule of thumb for avoiding a clearly unusable optimizer
scale. It does not support claims that one candidate is scientifically superior.
The chosen rate for an algorithm is then used for all three actor sizes. If this
rule changes, the change must be documented before any reported experiment run.

### Deterministic reference controller

The reference controller is a debugging aid for checking that the environment
can be driven without learning:

$$
A_t^{\mathrm{steer}}=
\operatorname{clip}(-0.15d_t-0.8\phi_{e,t}+50\bar\kappa_t,-1,1),
$$

$$
v_t^\star=min\left\{50,
\sqrt{12/\max(|\bar\kappa_t|,10^{-4})}\right\},
\qquad
A_t^{\mathrm{throttle}}=
\operatorname{clip}((v_t^\star-v_t)/6,-1,1).
$$

These constants are hand-designed project values, not results from the course
theory:

- `0.15` makes a one-metre lateral error contribute `0.15` steering;
- `0.8` gives a heading error in radians a stronger corrective effect;
- `50` makes curvature $0.01\,\mathrm{m^{-1}}$ contribute `0.5` steering;
- $12\,\mathrm{m\,s^{-2}}$ defines a curvature-dependent target through
  $v^2|\kappa|$, deliberately below the car's $20\,\mathrm{m\,s^{-2}}$ friction
  budget: preview curvature is averaged over the lookahead and so understates a
  corner on entry, and braking spends grip that is then unavailable to turn, so
  a controller aiming at the limit arrives at corners already beyond it;
- $50\,\mathrm{m\,s^{-1}}$ keeps the reference below the environment maximum;
  and
- division by `6` turns a $6\,\mathrm{m\,s^{-1}}$ speed error into saturated
  throttle or braking.

The controller is neither an expert demonstrator nor training data. Its purpose
is to expose a broken observation, reward or control convention before neural
learning is blamed. On twenty-four generated circuits it completes every lap,
averaging $15.9\,\mathrm s$, which is the standing evidence that the task is
solvable with the current physics and reward.

### Fixed circuit for Experiment 1

Generate 100 candidate circuits from dedicated deterministic identities. A
candidate is eligible when:

- at least 15% of its arc-length samples have
  $|\kappa|\le0.002\,\mathrm{m^{-1}}$, representing approximately straight
  sections with radius at least $500\,\mathrm m$; and
- at least 5% have $|\kappa|\ge0.01\,\mathrm{m^{-1}}$, representing materially
  curved sections with radius at most $100\,\mathrm m$.

These thresholds ensure the selected circuit contains both behaviours needed
for the racing question. A dry run found 85 eligible candidates out of 100, so
the rule is selective without being fragile.

For every eligible circuit, compute track length, median $|\kappa|$ and the 90th
percentile of $|\kappa|$. Center each feature on its eligible-set median and
divide it by its eligible-set interquartile range. Omit a feature only if that
range is zero. Select the circuit with the smallest Euclidean norm of the
resulting vector, breaking an exact tie by lower logical identity. Save the
complete circuit as `tracks/experiment_1.json`.

No learned return, completion or lap time can influence this selection.

### Physics version

The environment previously used a kinematic bicycle model with no lateral grip,
no drag, and no steering-rate limit. Under it, full throttle remained close to
optimal everywhere: at $70\,\mathrm{m\,s^{-1}}$ through the tightest corner of a
typical generated circuit the car pulled about $34g$, and a trained A2C policy
finished laps at the speed limit with the throttle open.

This protocol previously deferred the decision to a pre-registered trigger
measured on trained trajectories. That trigger is superseded: the diagnostic it
was meant to detect was observed directly, so the constraint was specified and
frozen rather than tested for. The grip-limited model in [`MDP.md`](MDP.md) is
the physics for both reported experiments. Its three additions are a shared tyre
friction budget of $20\,\mathrm{m\,s^{-2}}$, a steering rate limit of
$180°\,\mathrm s^{-1}$, and quadratic aerodynamic drag derived from the existing
speed and acceleration limits.

The unconstrained model remains selectable through configuration, so a later
ablation can report both. Any such comparison is development evidence unless it
is added to this protocol before the runs happen.

### Reduced-budget end-to-end validation

Before expensive runs, every unique algorithm/actor-size path and both
observation paths execute with a separate root and a budget of 40,000
interactions. This budget is not used to judge learning. It is large enough for
eight maximum-length REINFORCE episodes, so even the slowest collection boundary
can perform an update.

The purpose is only to prove that training, evaluation, checkpointing, resume,
run-output validation and analysis execute. Outputs are written under
`results/reduced_budget_end_to_end_validation/` and cannot be loaded as reported
experiment data.

## How reproducible randomness works

One integer written on a run specification is called its **root identity**. It
must control the whole run reproducibly, but using one mutable random generator
for everything would create accidental coupling. For example, adding an
evaluation could consume random numbers and change every later training action.

The solution is to derive independent child generators for distinct jobs:

- actor initialization;
- critic initialization;
- stochastic policy actions;
- environment reset;
- training-circuit schedule;
- PPO minibatch order; and
- evaluation or baseline-policy sampling where sampling exists; and
- track generation.

For a concrete example, reported Experiment 1 root `0` always derives the same
actor-initialization child and the same policy-sampling child. Evaluating that
actor more often consumes neither child, so the training trajectory does not
change. Root `0` used during pre-experiment configuration is different because
it belongs to a different namespace.

`numpy.random.SeedSequence` implements this hierarchy. The stable protocol key
`0`, namespace code and local identity are data—not secret randomness:

$$
\text{SeedSequence input}=[0,\text{namespace code},\text{local identity}].
$$

| Purpose | Namespace code | Local identities |
|---|---:|---|
| Experiment 1 reported roots | 1 | `0..4` |
| Experiment 2 reported roots | 2 | `0..4` |
| Learning-rate configuration | 3 | `0..2` |
| Capability check | 4 | `0..2` |
| Reduced-budget end-to-end validation | 5 | `0` |
| Controlled-problem algorithm validation | 6 | `0..4` |
| Experiment 1 circuit candidates | 7 | `0..99` |
| Multi-circuit development checks | 8 | `0..7` |
| Experiment 2 training circuits | 9 | one identity per saved training circuit |
| Experiment 2 validation circuits | 10 | `0..15` |
| Experiment 2 test circuits | 11 | `0..31` |
| Randomized execution order | 12 | one identity per experiment |

Within a run, stream indices `1..8` correspond in order to the eight jobs listed
above. A saved circuit uses its split namespace and circuit index as its local
identity, and the first `uint32` from stream `8` as the track-generator seed.
Frenet and LiDAR root $i$ use identically seeded stream `5` to select the same
ordered training-circuit identities even though their generator objects are
separate.

Every recorded run stores both human-readable logical identities and generated
integer states. Checkpoints retain all mutable generator states required for an
exact resume on the supported hardware and software stack.

## Execution hardware and timing

Neural-network training and evaluation use the available NVIDIA GeForce RTX
2060 with CUDA and PyTorch `float32`. Racing environments and their geometry
queries execute in persistent CPU worker processes: REINFORCE uses eight workers
to collect its eight complete episodes concurrently, while A2C and PPO start
with the number of physical CPU cores and store a pooled 2048-transition
rollout. The selected worker count is configurable and recorded. The main
process and every worker apply one PyTorch intra-op thread, one inter-op thread,
deterministic algorithms and disabled cuDNN benchmarking. CPU-only neural
execution remains valid for unit tests and reduced development checks, but it
is not comparable reported timing data.

All reported runs use the same physical GPU, CUDA/PyTorch build, driver, and
the same algorithm-specific worker and thread configuration. PyTorch
deterministic algorithms are enabled
with errors rather than warnings, and cuDNN benchmarking is disabled. PyTorch
does not guarantee identical results across releases, platforms or CPU/GPU
execution, which is why the exact stack is retained with every run. See the
[official reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html).

Timing categories do not overlap:

- environment collection time;
- actor/critic optimization time;
- deterministic evaluation time;
- checkpoint and metric-persistence time; and
- end-to-end time from accepted manifest to `completion.json`.

Training-only time is collection plus optimization. End-to-end time is also
reported so evaluation and persistence overhead remain visible. Reported runs
execute one at a time; no other training process competes for the GPU.

## Experiment 1 — Actor size on one circuit

### Question and hypotheses

> How does actor-network capacity affect final driving performance, interaction
> efficiency, convergence reliability and computational cost on one fixed
> circuit?

The actor-size comparison is repeated for REINFORCE, A2C+GAE and PPO. The
algorithm comparison is secondary: it describes the practical effect of adding
a critic, GAE and bounded sample reuse, but it does not assume the more elaborate
algorithm must win.

- **Capacity:** actor size can change final task performance; larger is not
  assumed to be better.
- **Efficiency:** actor size can change interactions and computation required to
  reach the task threshold.
- **Reliability:** actor size can change the fraction of roots that learn a
  stable lap-completing policy.
- **Algorithm:** A2C+GAE and PPO are expected to reduce variance or improve
  sample use relative to REINFORCE, but contrary evidence remains valid.

### Experimental units and design matrix

One independent unit is a complete training run identified by

$$
(\text{algorithm},\text{actor size},\text{root identity}).
$$

| Algorithm | `(32, 32)` actor | `(64, 64)` actor | `(256, 256)` actor |
|---|---:|---:|---:|
| REINFORCE | 5 roots | 5 roots | 5 roots |
| A2C+GAE | 5 roots | 5 roots | 5 roots |
| PPO | 5 roots | 5 roots | 5 roots |

The complete experiment has $3\times3\times5=45$ runs. Root identities `0..4`
come from the Experiment 1 reported-results namespace and are paired across
actor sizes and algorithms.

### Fixed conditions and training

Every run uses:

- the saved `tracks/experiment_1.json` circuit and canonical start;
- Frenet observation $(d_t,\phi_{e,t},v_t,\bar\kappa_t)$;
- the same action mapping, reward, episode limit and frozen physics version;
- the bounded Gaussian policy and optimizer contract in `LEARNING.md`;
- one fixed `(64, 64)` critic for A2C and PPO;
- the learning rate selected before the experiment for its algorithm;
- 2,000,000 training interactions;
- deterministic evaluation every 50,000 training interactions; and
- checkpoints every 250,000 interactions and at the final budget.

The 2,000,000-interaction budget is a project planning value. It provides 40
evaluation positions and corresponds to at most 400 full-length episodes. It is
not derived from theory; changing it requires a dated amendment based only on
pre-experiment runtime or capability evidence and must affect every one of the
45 run specifications.

One interaction is one call to `RacingEnv.step`, regardless of four internal
physics substeps. Evaluation interactions are counted separately and never
enter the training budget. The final policy at exactly the common budget is the
primary result; a best-seen checkpoint is only a diagnostic.

### Deterministic evaluation and convergence

Evaluation uses
$A_t^{\mathrm{eval}}=\tanh(\boldsymbol\mu_{\mathbf\theta}(O_t))$ from the
canonical start with zero speed, even though training samples its start pose.
Holding the evaluation start fixed keeps every reported number an answer to the
same question and reproducible from the seed alone. It does not change model
parameters, optimizers, observation statistics, training RNGs or checkpoint
selection.

One evaluation episode per checkpoint is sufficient and is what runs. Both the
policy and the start are deterministic and the circuit is fixed, so repeated
episodes are identical; earlier runs took sixteen of them and reported a
standard deviation that was exactly zero at every checkpoint.

Stable convergence is the first of three consecutive evaluations that complete
the lap in at most $32$ simulated seconds, twice the reference controller's
$15.9\,\mathrm s$ average. A run that never meets the rule is right-censored at
2,000,000 interactions and remains in every success-rate and learning-curve
summary.

Evaluation records the episode outcome, not only its return. A return alone
cannot distinguish crashing from idling from lapping slowly, and the outcomes
now include `stalled` for a car that has stopped making progress.

### Data recorded for every run

The following are the actual tracked quantities for Experiment 1.

**Identity and reproducibility**

- algorithm, actor widths and exact actor parameter count;
- critic widths and parameter count where applicable;
- root identity and every derived seed state;
- complete environment, model, optimizer and evaluation configuration;
- protocol revision, manifest checksum, git commit and dirty state;
- Python and dependency freeze, OS, CPU, GPU, driver, CUDA and PyTorch versions;
  and
- start/end timestamps and run completion state.

**Every training and evaluation episode**

- discounted training target totals where relevant and raw undiscounted return;
- agent steps and elapsed simulated time;
- explicit finish, crash or time-limit outcome;
- final and maximum normalized progress;
- lap time when completed;
- mean, standard deviation, extrema and selected speed quantiles;
- mean and extrema of throttle/brake and absolute steering; and
- fractions of positive throttle, braking and near-saturated steering.

The absolute-steering threshold defining “near-saturated” is an explicit
reported-run configuration value. The training command refuses to start a
reported run when it is absent; Step 10 freezes the value before data collection.

**Every optimizer update**

- actor loss, learning rate, entropy proxy and learned log-standard-deviation;
- actor gradient norm before clipping, weight norm and update norm;
- critic loss, gradient norm, weight norm, predictions, targets, advantages and
  explained variance for A2C and PPO;
- importance-ratio distribution, clip fraction and approximate KL for PPO; and
- interaction count and optimization duration.

**Computational cost**

- collection, optimization, evaluation, persistence and end-to-end times;
- completed episodes and optimizer updates;
- interactions per collection second and per training second;
- actor, critic and total parameter counts; and
- peak process and GPU memory where available.

Selected deterministic trajectories at every 250,000 interactions and at the
final budget retain progress, pose, current/preview curvature, speed, controls,
reward, outcome and $v^2|\kappa|$. Step-level training trajectories are not
stored because their I/O would distort timing.

### Outcomes and analysis

Final actor-size conclusions use outcomes in this order:

1. number and fraction of roots whose final policy completes the lap;
2. completed lap time, always with its completion denominator;
3. final return and normalized progress for every root; and
4. number and fraction of final policies that crash.

Learning and resource outcomes are:

- evaluation return and progress curves against training interactions;
- normalized area under each evaluation curve;
- interactions, episodes and wall time to stable convergence;
- convergence fraction and censoring state;
- final-window regression or instability;
- collection throughput, optimization time, end-to-end time and peak memory;
  and
- optimization diagnostics listed above.

For each algorithm/size cell, report all five root values, mean and sample
standard deviation, median and interquartile range, minimum/maximum, and a 95%
bootstrap interval resampling roots. Actor-size and secondary algorithm
comparisons use paired within-root differences. With only five roots, intervals
are descriptive; conclusions emphasize magnitudes and raw outcomes.

Normalized curve area is the trapezoidal area from the first recorded
evaluation to the final evaluation divided by that recorded interaction span.
The five-root confidence interval exhaustively evaluates all $5^5$ root
bootstrap resamples, so it has no sampling seed. Final-window diagnostics use
the last three evaluations and report the change, range and final-minus-best
value; no undeclared binary instability threshold is introduced. A
representative final trajectory is the root closest to its cell median final
return, with lower root identity breaking an exact tie.

Non-converged runs receive the full budget only in the explicitly named
restricted-time summary and remain marked as censored. Lap time is never
averaged after silently removing failures.

Required Experiment 1 outputs are:

- complete configuration and parameter-count table;
- final task metrics and raw root points for all nine cells;
- completion/crash counts;
- evaluation learning curves and normalized curve areas;
- convergence interactions/time with censoring;
- performance and cost versus actor parameter count;
- throughput, memory and end-to-end runtime;
- optimization diagnostic panels; and
- curvature-conditioned speed, throttle and steering for representative final
  policies selected by a predeclared rule.

### PPO actor selection for Experiment 2

Only Experiment 1 PPO results choose the Experiment 2 actor:

1. find the size with highest mean final deterministic return;
2. compute paired root-level return deficits for every other size;
3. admit a size whose mean deficit is no greater than one standard error and
   whose completion count is no more than one root below the best size; and
4. choose the admitted actor with the fewest parameters.

An operationally corrupted run is repaired and rerun under its original
specification. A valid poor run is never replaced. Record the calculation before
examining any Experiment 2 test circuit.

### Experiment 1 limitations

- Five roots give only a modest estimate of training variation.
- One circuit makes conclusions circuit-specific.
- Width changes at fixed two-layer depth do not cover every notion of network
  complexity.
- The fixed critic isolates actor width but may constrain the largest actor.
- Algorithm differences cannot be attributed only to abstract complexity
  because their estimators and update schedules differ.

## Experiment 2 — Circuit generalization and observation choice

### Question and hypotheses

> How well does the selected PPO actor generalize from procedurally generated
> training circuits to unseen circuits, and how does Frenet observation compare
> with local LiDAR sensing?

- **Generalization:** PPO trained over generated circuits is expected to retain
  useful performance on unseen generator seeds.
- **Observation information:** Frenet is expected to learn faster because it
  directly exposes track-relative geometry and preview curvature; LiDAR may have
  a larger efficiency or final-performance gap.
- **Track variation:** both conditions can vary substantially across held-out
  circuit geometry, so per-circuit outcomes accompany root-level summaries.

### Experimental units and design matrix

One independent training unit is

$$
(\text{observation type},\text{root identity}).
$$

| Algorithm | Actor | Observation | Roots |
|---|---|---|---:|
| PPO | Selected by Experiment 1 rule | Frenet | 5 |
| PPO | Same hidden widths | LiDAR | 5 |

The complete experiment contains ten training runs. Experiment 2 reported root
identities are `0..4`. Within each root, the Frenet and LiDAR runs are paired by
training-circuit schedule, budget, PPO settings, validation circuits and test
circuits. Their mutable RNG objects remain separate.

### Circuit splits and observations

All circuits use the same frozen generator but disjoint deterministic
namespaces:

- eight development circuits used only before the reported experiment;
- an unbounded training schedule indexed by root and episode;
- 16 fixed validation circuits with logical identities `0..15`; and
- 32 fixed test circuits with logical identities `0..31`.

Sixteen validation circuits balance evaluation cost with varied geometry at 40
learning-curve positions. Thirty-two test circuits provide a broader final
distribution. These counts are explicit project resource choices, not a formal
power calculation. Every generated integer seed and geometry summary is stored.

The Frenet actor and critic receive

$$
O_t^{\mathrm{Frenet}}=(d_t,\phi_{e,t},v_t,\delta_t,\bar\kappa_t).
$$

The LiDAR actor and critic receive

$$
O_t^{\mathrm{LiDAR}}=(v_t,\delta_t,\widetilde r_t^{(1)},\ldots,
\widetilde r_t^{(16)}).
$$

Both conditions carry the speed and the steering angle, which are vehicle state
rather than perception. Only the track representation differs, which is the
comparison the experiment is about.

Neither critic receives privileged information. Each condition learns its own
observation-normalization statistics from training only and freezes them during
evaluation. LiDAR remains feed-forward without frame stacking, so its partial
observability is part of the interpretation.

### Fixed conditions and training

Every run uses:

- PPO and the actor widths selected before any Experiment 2 test result;
- identical hidden widths and fixed `(64, 64)` critic widths;
- the same PPO optimizer, GAE and clipped-loss configuration;
- the same action mapping, reward, episode limit and frozen physics version;
- the canonical start on every circuit;
- the same logical sequence of training circuits for a paired root;
- 2,000,000 training interactions;
- deterministic validation every 50,000 interactions; and
- checkpoints every 250,000 interactions and at the final budget.

Circuits change only at episode reset. Different observation policies can have
different episode lengths and therefore encounter different numbers of circuits
within the common interaction budget, but episode index $e$ always maps to the
same circuit identity for paired root $i$. Circuit exposure is recorded rather
than silently treated as identical.

Validation and test interactions do not enter the training budget. Validation
does not update networks, optimizers, normalizers, training-circuit schedules or
training RNGs. Test circuits cannot influence training, learning-rate choice,
actor-size selection, convergence or checkpoint selection.

### Deterministic evaluation and convergence

Every validation checkpoint evaluates the deterministic policy once on each of
the 16 validation circuits. Stable convergence is the first of three consecutive
checkpoints with:

- validation completion rate at least `0.75`; and
- median validation normalized progress at least `0.95`.

These thresholds require completion on at least 12 of 16 circuits while keeping
typical progress near a full lap. They are project definitions, not values from
the PPO theory. Runs that never meet them are censored at the common budget.

The final policy is evaluated once on fixed training-reference circuits, all 16
validation circuits and all 32 unseen test circuits. The test set is opened only
after training and selection are complete.

### Data recorded for every run

Experiment 2 records the same identity, optimizer and computational diagnostics
listed for Experiment 1, with the following circuit-specific fields made
mandatory.

**Every training episode**

- observation condition, root identity and training episode index;
- circuit logical identity, generated seed and split;
- return, steps, simulated time and finish/crash/time-limit outcome;
- final and maximum progress, lap time when completed;
- speed and control summaries; and
- current training interaction count.

**Every validation or final evaluation circuit**

- checkpoint interaction count and circuit identity;
- circuit length, curvature quantiles and other frozen geometry summaries;
- return, completion, crash, progress and conditional lap time;
- observation-normalizer state checksum; and
- selected step-level trajectory fields for geometric diagnosis.

**Every PPO update and computational interval**

- actor/critic losses, targets, advantages and explained variance;
- entropy proxy, learned dispersion, approximate KL, clip fraction and
  importance-ratio distribution;
- gradient, weight and update norms;
- collection/optimization/evaluation/persistence durations;
- throughput, parameter counts and peak CPU/GPU memory; and
- number and identities of training circuits encountered so far.

### Outcomes and analysis

For each root, first aggregate final test performance across its 32 test
circuits. The primary Frenet-minus-LiDAR comparison then uses five paired
root-level differences. Individual circuits reveal where a representation helps
or fails but are not treated as 32 independently trained policies.

Primary final outcomes are:

1. completion rate across test circuits;
2. crash rate across test circuits;
3. normalized progress and deterministic return for every test circuit;
4. completed lap time with its completion denominator; and
5. the distribution of these outcomes across circuit geometry.

Generalization and efficiency outcomes are:

- training-reference, validation and test performance;
- validation-to-test and training-reference-to-test gaps;
- validation learning curves against training interactions;
- interactions and time to the validation convergence threshold;
- completion, crash and progress stratified by predeclared length and curvature
  bins;
- per-circuit paired Frenet-minus-LiDAR differences; and
- runtime, throughput, memory and PPO optimization diagnostics.

Report raw root-level values, mean/sample standard deviation, median/interquartile
range, extrema and a 95% paired bootstrap interval resampling the five root-level
differences. Track distributions are additional descriptive evidence. A small
generalization gap accompanied by poor absolute performance is not successful
generalization.

Required Experiment 2 outputs are:

- the recorded PPO actor-size calculation;
- complete split manifest and circuit-geometry summary;
- held-out completion, crash, progress, return and conditional lap-time table;
- paired root-level Frenet-minus-LiDAR differences;
- per-circuit paired distributions and geometry-stratified plots;
- training-reference, validation and test gaps;
- validation learning curves and convergence censoring;
- computation, memory and optimization diagnostics; and
- explicit counts for every denominator.

### Experiment 2 limitations

- Five training roots provide modest evidence about optimizer randomness.
- Test circuits cover only the frozen procedural generator distribution.
- Feed-forward LiDAR is intentionally partially observable and may be
  disadvantaged relative to a recurrent sensor policy.
- Different episode lengths produce different total circuit exposure within an
  equal interaction budget.
- Input dimensions create a small unavoidable parameter-count difference even
  with equal hidden widths.

## Run-output and interpretation rules

### Directory layout

The directory names state why the run exists:

```text
results/
  pre_experiment_configuration/
    <purpose>/<run-id>/
  reduced_budget_end_to_end_validation/
    <experiment-path>/<run-id>/
  reported_experiments/
    experiment_1/<run-id>/
    experiment_2/<run-id>/
```

Every run directory contains:

```text
manifest.json
config.json
metadata.json
episodes.jsonl
updates.jsonl
evaluations.jsonl
checkpoints/
trajectories/
completion.json
```

The analysis-complete recording contract is schema version 2. Earlier
development runs are intentionally rejected because they lack checkpoint timing,
circuit geometry, resource summaries or retained learned-policy trajectories.

The schema rejects loading pre-experiment or reduced-budget results as reported
experiment data. Aggregated tables record input run identifiers and checksums.
Routine result trees and large checkpoints are not committed; frozen manifests,
compact validation fixtures and final summary tables may be.

### Failure and reporting rules

- An exception, invalid run output or hardware interruption is an operational
  failure. Repair the cause and rerun the same specification.
- A valid run that crashes, learns poorly or never converges is a scientific
  outcome and remains in analysis.
- The final fixed-budget policy is primary; best-seen performance is labelled as
  diagnostic.
- Completed-only lap-time summaries always state completion counts.
- Convergence summaries show success fractions and censoring instead of
  averaging only successful runs.
- Learning curves align on training interactions, never episodes or updates.
- Smoothing is display-only and never changes convergence or area calculations.
- Actor-size claims are made within each algorithm before secondary algorithm
  comparisons.
- Experiment 2 test results cannot alter its actor, training or normalization.
- Negative, high-variance and non-monotonic results are retained.

Every table and figure is regenerated from raw run records. Console output and
manually copied summary values are not authoritative.

Each evaluation record stores cumulative collection and optimization time, so
time to convergence is read at the first qualifying checkpoint rather than
estimated from final runtime. Completion records contain the full resource and
timing summary. Circuit outcomes carry frozen length and curvature summaries,
and selected trajectory documents carry pose, speed, controls, current and
preview curvature, outcome and $v^2|\kappa|$. Consequently, analysis needs no
checkpoint loading, console output or directory-name inference.
