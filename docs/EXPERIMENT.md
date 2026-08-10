# Experimental Protocol

## Purpose and Status

This file specifies the scientific experiments for the racing project. 
It defines what will be compared, what will remain fixed, what will be measured and how conclusions will be drawn.

Three kinds of run must remain visibly distinct:

- **pilot runs** select or validate settings using pilot-only roots;
- **smoke runs** prove that configurations execute at deliberately small
  budgets; and
- **measurement runs** are the only runs used for the reported experimental
  conclusions.

Pilot and smoke results are retained, but never pooled with measurement results.
The protocol revision and frozen manifest checksum used by every measurement
run must be stored in its metadata.

## Research Questions

### Experiment 1 — Policy Size on One Circuit

The primary research question is:

> How does the capacity of the policy network affect final driving performance,
> sample efficiency, convergence reliability and computational cost when
> learning one fixed circuit?

The policy-size comparison is repeated independently for three progressively
more sophisticated policy-gradient algorithms:

1. REINFORCE;
2. A2C with Generalized Advantage Estimation; and
3. Proximal Policy Optimization.

The algorithm comparison is secondary. It checks whether variance reduction,
bootstrapping and controlled sample reuse produce the expected practical gains,
but a negative or non-monotonic result is valid evidence. “More complex” is not
treated as a guarantee that an algorithm must win.

#### Hypotheses

- **H1 — Capacity:** policy size affects final task performance. Increasing
  capacity may help up to a point, but diminishing or negative returns are
  possible.
- **H2 — Efficiency trade-off:** larger policies require more computation per
  update and may require a different number of environment interactions to
  reach the same task threshold.
- **H3 — Reliability:** policy size affects the between-seed probability of
  learning a stable lap-completing policy.
- **H4 — Algorithm:** A2C+GAE and PPO are expected to be more sample-efficient
  or reliable than actor-only REINFORCE, with PPO expected to benefit from
  bounded sample reuse. This is a secondary descriptive comparison rather than
  the main policy-space question.

### Experiment 2 — Circuit Generalization and Observation Choice

The second research question is:

> How well does the selected PPO policy generalize from procedurally generated
> training circuits to unseen circuits, and how does that behaviour differ
> between Frenet and LiDAR observations?

Experiment 2 compares two observation conditions while keeping the algorithm,
policy architecture, training-track schedule, interaction budget and held-out
tracks paired.

#### Hypotheses

- **H5 — Generalization:** PPO trained on a distribution of generated circuits
  will retain useful performance on circuit seeds not used for training or
  calibration.
- **H6 — Observation information:** Frenet observations are expected to learn
  faster because they expose compact track-relative geometry and preview
  curvature. LiDAR exposes lower-level local geometry and may therefore have a
  larger sample-efficiency or final-performance gap.
- **H7 — Track variation:** both observation conditions will exhibit
  track-dependent variability; differences must therefore be reported across
  held-out tracks as well as across training seeds.

## Decisions Already Fixed

The following decisions are part of the approved design:

| Concern | Decision |
|---|---|
| Experiment 1 algorithms | REINFORCE, A2C+GAE and PPO |
| Policy-size factor | Small `(32, 32)`, medium `(64, 64)`, large `(256, 256)` hidden layers |
| Meaning of size | Actor/policy MLP only; report exact actor parameter count |
| Critic capacity | One fixed architecture across all actor sizes within A2C and PPO |
| Experiment 1 circuit | One saved procedurally generated circuit |
| Experiment 1 observation | Frenet |
| Measurement roots | Five paired root-seed identities per condition |
| Budget fairness | Equal training-environment interactions within a comparison |
| Evaluation policy | Deterministic transformed mean action |
| Experiment 2 algorithm | PPO |
| Experiment 2 size | Selected from Experiment 1 by the rule below |
| Experiment 2 circuits | Procedurally generated training circuits and disjoint fixed held-out circuits |
| Experiment 2 observations | Frenet versus 16-ray LiDAR |
| Observation fairness | Actor and critic receive the same condition-specific observation; no privileged critic |
| Physics order | First diagnose version 0; add grip before measurement only if the pilot trigger is met |
| Reward | Hold fixed across policy sizes, algorithms, observations and the before/after grip pilot |

The five measurement roots will have stable identities `0` through `4` within a
dedicated measurement namespace. Pilot, smoke, track and evaluation streams use
different namespace identifiers even if their local integer labels match.

## Values to Lock During Phase 2

The following values depend on implementation validation or runtime evidence.
They must be written here and into the final manifests before measurement starts;
they may not be chosen by individual runs:

- the exact bounded Gaussian parameterization, activation, initialization and
  initial dispersion;
- the fixed critic architecture;
- discount, GAE, normalization, entropy and gradient-clipping choices;
- optimizer and update settings for each algorithm;
- the common training-interaction budget and evaluation/checkpoint cadences;
- the fixed Experiment 1 track seed and geometry;
- pilot roots and algorithm-calibration allowance;
- the reference controller and numerical convergence threshold;
- the capable-policy gate and quantitative lateral-grip trigger;
- the final physics version and, if applicable, grip constant;
- Experiment 2 validation/test track counts and their seed lists;
- environment worker count, thread settings and timing boundary; and
- artifact retention and trajectory-sampling cadence.

These are preregistration fields, not invitations to tune after measurement.
Phase 2 may amend them using only pilot results. The final acceptance commit
must replace this section with a dated locked-values table or link each field to
an immutable machine-readable manifest.

## Common Experimental Protocol

### Experimental Units and Pairing

For Experiment 1, the independent experimental unit is one complete training
run identified by:

$$
(\text{algorithm},\ \text{actor size},\ \text{root seed}).
$$

There are $3\times3\times5=45$ measurement runs. Because the fixed-circuit
environment and deterministic evaluation policy have no evaluation-time noise,
repeating the same checkpoint evaluation from the same start state would return
the same trajectory. Repeated evaluation episodes are therefore not treated as
independent evidence. Variation is estimated across the five training roots.

For Experiment 2, the independent training unit is:

$$
(\text{observation type},\ \text{root seed}).
$$

Frenet and LiDAR runs are paired by root seed and receive the same logical
training-track schedule. Held-out tracks are repeated evaluation cases nested
within a training run. Track episodes are reported individually, but inference
about observation choice first aggregates held-out performance within each root
and then compares the five paired root-level summaries. This avoids pretending
that many tracks generated by one learned policy are many independently trained
policies.

### Reproducible Randomness

One explicit root produces independent named streams for:

- actor initialization;
- critic initialization;
- policy action sampling;
- environment reset;
- procedural training-track selection;
- minibatch permutation; and
- evaluation or reference-policy sampling where applicable.

Changing evaluation cadence must not change the training trajectory. Frenet and
LiDAR pairs in Experiment 2 share track identities and scheduling but do not
share mutable RNG objects. Checkpoints retain all stream states required for an
exact resume on the supported machine.

### Environment Interactions

One environment interaction is one agent action passed to `RacingEnv.step`,
regardless of the four internal physics substeps. Training, pilot, smoke and
evaluation interactions are counted separately.

The training budget and convergence x-axis use training interactions, not
updates or episodes. Episode counts are retained as a secondary description
because crashes and lap times change episode length. Evaluation interactions do
not count toward sample efficiency.

### Evaluation

Evaluation uses the deterministic action defined in `docs/LEARNING.md`, with no
policy sampling. It must not:

- update actor, critic or optimizer state;
- update observation-normalization statistics;
- consume a training RNG stream; or
- select a checkpoint using held-out test results.

The primary final result is the policy at the fixed end of the training budget,
not the best checkpoint discovered after the fact. Best-seen performance may be
reported as a clearly marked diagnostic, never as the primary final score.

Experiment 1 evaluates on its one canonical fixed-track start. Experiment 2
evaluates the final policy on fixed training-reference, validation and test
track lists. Each deterministic policy/track pair needs one episode unless a
future protocol amendment adds evaluation-time randomness.

### Common Controls

Within each comparison, the following remain fixed unless explicitly named as a
factor:

- environment action and reward;
- episode limit and initial-state rule;
- physics version;
- track generator configuration;
- training interaction budget;
- evaluation cadence and deterministic action rule;
- observation normalization method;
- logging instrumentation;
- machine, device, worker and thread settings; and
- software dependency versions.

Run order should be blocked by root and randomized within each block using a
recorded scheduling seed. This reduces systematic timing bias from machine load,
thermal state or run order. No two timed measurement runs should compete for the
same CPU or accelerator unless concurrency itself is fixed as part of the
protocol.

## Preparatory Pilots and Freezing Procedure

Pilots exist to make the experiment executable, not to provide extra favourable
observations.

### Fixed-Circuit Selection

The Experiment 1 circuit is chosen before policy measurement by a geometry-only
procedure:

1. generate a finite, predeclared set of candidate seeds with the approved track
   generator configuration;
2. reject only candidates that fail the existing geometric validator;
3. summarize length and the distribution of absolute curvature;
4. choose a circuit with both straights and materially curved sections using
   thresholds locked before inspecting learned-policy results; and
5. save the exact JSON circuit as `tracks/experiment_1.json`.

No learned return, completion or lap time may influence circuit selection.

### Algorithm Calibration

Each algorithm may use settings appropriate to its update rule. Calibration:

- uses only the medium `(64, 64)` actor;
- uses pilot roots disjoint from measurement roots;
- gives candidate settings the same pilot interaction allowance within an
  algorithm;
- evaluates against the same fixed track and pilot protocol;
- chooses settings using a predeclared task score and stability diagnostics;
  and
- freezes one algorithm configuration for all three actor sizes.

This prevents a separate hyperparameter search from making each network size a
different treatment. The secondary algorithm comparison is interpreted with
the caveat that algorithms necessarily have different update structures and may
use different calibrated optimizer settings.

### Version-0 Physics and Grip Decision

Before the Experiment 1 measurement matrix, a designated PPO pilot is trained on
version-0 physics. Cornering is interpreted only if it passes a capable-policy
gate, such as stable useful progress or lap completion, whose exact threshold is
locked before the pilot.

The analysis uses bins derived from the fixed track's absolute curvature, not
from whichever portions the policy happens to visit. For every bin it reports:

- visitation count and fraction;
- speed and maximum speed;
- throttle/brake and steering;
- crash and incomplete-episode contribution; and
- the lateral-acceleration proxy $v^2|\kappa|$.

The grip trigger must combine a documented physical limit with predeclared
behavioural evidence. In outline, grip is triggered only when a capable policy
repeatedly exceeds the physical cornering limit in materially curved sections
and does not produce the predeclared reduction in throttle or speed from low- to
high-curvature bins. The physical bound, required count and reduction tolerance
are Phase-2 lock values.

If the trigger is not met, version 0 is retained and the negative decision is
recorded. If it is met:

1. specify the minimum grip model and constants in `docs/MDP.md`;
2. implement it while keeping version 0 selectable;
3. repeat the same pilot with the same reward and observation; and
4. freeze the grip-limited version only after the policy remains capable and the
   intended speed-versus-curvature constraint is observable.

The full 45-run matrix is run once, on the frozen physics version. The no-grip
pilot is not expanded into a second full matrix unless a later, separately
approved experiment asks a scientific question about dynamics versions.

## Experiment 1

### Design Matrix

| Algorithm | Small actor | Medium actor | Large actor |
|---|---:|---:|---:|
| REINFORCE | 5 roots | 5 roots | 5 roots |
| A2C+GAE | 5 roots | 5 roots | 5 roots |
| PPO | 5 roots | 5 roots | 5 roots |

The hidden sizes are:

- small: `(32, 32)`;
- medium: `(64, 64)`; and
- large: `(256, 256)`.

Only the actor hidden sizes change across the columns. For A2C and PPO, the
critic hidden sizes remain fixed. The input/output layers and exact actor
parameter count depend on the Frenet observation and two-dimensional action and
must be logged.

### Fixed Conditions

- one saved circuit selected by the geometry-only procedure;
- Frenet observations;
- canonical starting state;
- one frozen reward and physics version;
- one common training-interaction budget;
- one evaluation/checkpoint schedule;
- five paired root identities;
- one fixed critic architecture for A2C and PPO; and
- algorithm-specific settings frozen with the medium actor before measurement.

### Primary Outcomes

Policy-size conclusions use task outcomes in this order:

1. **lap-completion reliability:** number and fraction of training roots whose
   final deterministic policy completes the circuit;
2. **lap time:** final deterministic lap time, always accompanied by its
   completion denominator;
3. **final return and normalized progress:** defined for completed, crashed and
   truncated episodes; and
4. **crash probability:** number and fraction of final policies ending off
   track.

Lap time must never be averaged after silently discarding non-completing roots.
When few policies complete, return and progress describe failures, while the
completion count remains the main task statement.

### Learning and Convergence Outcomes

- training and deterministic-evaluation learning curves against environment
  interactions;
- area under the evaluation learning curve over the common interaction budget;
- first interaction count at stable convergence;
- episodes and wall-clock time at the same convergence point;
- convergence success fraction within budget; and
- final-window instability or regression as a diagnostic.

Stable convergence is the first of a predeclared number of consecutive
evaluation checkpoints that both completes the lap and meets the fixed
performance threshold. Phase 2 locks the threshold and required consecutive
checkpoints using references and pilots. Runs that never converge are
right-censored at the common budget; they remain in success-rate and curve
summaries.

For comparable scalar summaries, define restricted time to convergence as the
observed convergence time for a successful run and the full budget for a
right-censored run. Report it with the convergence-success fraction and censor
markers; it is a lower bound on the unknown time a failed run would actually
need. Successful-only convergence summaries may also be shown, but must state
their denominator.

Area under the learning curve is reported as a threshold-independent sample
efficiency measure so a fragile threshold does not carry the entire conclusion.

### PPO Size Selection for Experiment 2

The selected PPO actor is determined only from Experiment 1 measurement results:

1. identify the PPO size with the highest mean final deterministic return;
2. for every other size, compute the five paired root-level differences from
   that best size and their standard error; admit a size when its mean deficit
   is no greater than one standard error and its completion count is no more
   than one root below the best completion count;
3. choose the smallest actor by parameter count from that set; and
4. if a required value is missing because a run failed operationally, repair
   and rerun that same specification before selection rather than changing the
   rule.

This is a smallest-adequate-network rule. It prevents Experiment 2 from always
inheriting the largest policy when its improvement is small relative to
between-seed uncertainty. The selected size and calculation are recorded before
any Experiment 2 test-track result is examined.

## Experiment 2

### Design Matrix

| Algorithm | Selected actor | Observation | Measurement roots |
|---|---|---|---:|
| PPO | Experiment 1 rule | Frenet | 5 |
| PPO | Experiment 1 rule | LiDAR | 5 |

The complete matrix contains ten training runs. Each Frenet/LiDAR pair shares:

- the same root identity;
- the same logical sequence of training circuit seeds;
- the same PPO and critic settings;
- the same training interaction budget and evaluation cadence;
- the same generator configuration; and
- the same validation and test circuits.

The observation input dimension changes the number of input-layer parameters.
Hidden sizes remain identical, and exact actor/critic parameter counts are
reported so the small unavoidable parameter-count difference is visible.

### Track Splits

All splits use the same frozen generator configuration but disjoint seed
namespaces:

- **pilot tracks:** used only while validating multi-track orchestration or
  locking shared PPO settings;
- **training tracks:** selected dynamically per episode from a deterministic
  schedule;
- **validation tracks:** fixed circuits used for protocol checks and learning
  curves, never for gradient updates; and
- **test tracks:** fixed unseen circuits used only for final reported
  generalization.

The exact validation and test counts and seed lists are locked during Phase 2.
The training schedule may be logically unbounded; the run records every circuit
actually used. Procedural generation and geometry caching must not change the
logical seed schedule.

No test circuit can be used for:

- PPO or normalization updates;
- hyperparameter calibration;
- actor-size selection;
- convergence-threshold selection; or
- deciding when to stop a run.

### Observation Conditions

The Frenet condition observes the four values specified in `docs/MDP.md`:

$$
(d_t,\phi_{e,t},v_t,\bar\kappa_t).
$$

The LiDAR condition observes speed and 16 normalized ranges:

$$
(v_t,\tilde r_t^{(1)},\ldots,\tilde r_t^{(16)}).
$$

Each condition fits its own running normalization statistics from its training
stream and freezes them for evaluation. This is required because the raw
features differ. The normalization algorithm and update rule remain the same.
The LiDAR policy is feed-forward; recurrence and frame stacking are outside the
experiment and must be acknowledged when interpreting its partial observability.

### Primary Outcomes

Final held-out test performance includes:

- completion rate across test circuits;
- crash rate across test circuits;
- normalized progress for every test circuit;
- deterministic return for every test circuit;
- lap time with completion denominator; and
- distribution of outcomes across track seeds.

For each root, metrics are first aggregated over test circuits. Frenet-minus-
LiDAR paired differences are then computed across the five roots. Per-track
paired differences are also plotted to reveal circuits on which an observation
helps or fails, but they are not substituted for independent training seeds.

### Generalization Outcomes

- training-reference, validation and test performance under the final policy;
- validation-to-test and training-reference-to-test gaps;
- completion and crash variation across circuit geometry;
- performance stratified by track length and curvature summaries fixed before
  test analysis; and
- learning curves on the fixed validation set versus training interactions.

Experiment 2 convergence uses a Phase-2-locked aggregate validation score and
the same consecutive-checkpoint principle as Experiment 1. Test circuits never
participate in convergence detection.

The generalization gap is always reported alongside absolute test performance.
A small gap caused by uniformly poor driving is not evidence of successful
generalization.

## Measures and Logging

The project will collect broad diagnostics, but distinguish scientific outcomes
from debugging signals.

### Run Identity and Provenance

Every run stores:

- experiment, run-kind and protocol revision;
- algorithm, actor size, observation and physics version;
- root and all derived seed identities;
- complete environment, model, algorithm and evaluation configurations;
- fixed track or track-split manifest;
- git commit and dirty status;
- dependency freeze and Python version;
- operating system, processor, accelerator and memory description;
- device, numerical dtype, worker count and thread settings; and
- start/end timestamps and completion status.

### Episode Metrics

For every training and evaluation episode:

- return;
- episode agent steps and elapsed simulated time;
- terminated/truncated outcome;
- lap completion, crash and timeout indicators;
- final and maximum normalized progress;
- lap time when completed;
- track seed and split;
- mean, standard deviation, extrema and selected quantiles of speed;
- mean and extrema of throttle/brake and absolute steering;
- fraction of positive throttle, braking and near-saturated steering; and
- grip activations or lateral-acceleration proxy when applicable.

### Update Metrics

Shared optimization diagnostics include:

- actor loss;
- learning rate;
- policy entropy or the approved transformed-distribution proxy;
- learned log-standard-deviation statistics;
- actor gradient norm before any clipping;
- actor weight norm;
- actor parameter-update norm; and
- update duration and cumulative training interactions.

A2C and PPO also record:

- critic loss;
- critic gradient and weight norms;
- value prediction and target statistics;
- advantage statistics; and
- explained variance.

PPO additionally records:

- approximate KL divergence;
- clip fraction;
- importance-ratio mean, standard deviation and extrema;
- entropy term and value term contributions; and
- actual number of completed epochs if a KL stop is enabled.

Weight norms are optimization diagnostics, not task-performance measures. A
larger norm is neither success nor failure by itself.

### Interaction and Computational Cost

Every run records:

- total training, pilot and evaluation interactions separately;
- completed training episodes and optimizer updates;
- environment collection time;
- optimization time;
- checkpoint/logging time;
- deterministic evaluation time;
- total end-to-end wall time;
- environment interactions per collection second;
- updates and interactions per total training second;
- wall time and interactions at convergence;
- actor, critic and total parameter counts; and
- peak process memory and peak accelerator memory when available.

Timing instrumentation and logging cadence remain identical across compared
conditions. Both training-only time and end-to-end time are reported so frequent
evaluation or checkpoint I/O is not confused with algorithm computation.

### Cornering Diagnostics

Selected evaluation trajectories retain step-level:

- wrapped and unwrapped progress;
- position and heading;
- current and preview curvature;
- speed;
- throttle/brake and steering;
- reward and lifecycle outcome; and
- grip activation or $v^2|\kappa|$.

Step-level training trajectories are not retained wholesale because their I/O
volume would alter runtime. The trajectory sampling cadence is frozen and equal
across conditions.

## Aggregation and Uncertainty

Raw seed-level values are always shown or made available. For each cell report:

- mean and sample standard deviation;
- median and interquartile range;
- minimum and maximum or individual seed points; and
- a 95% bootstrap confidence interval using the training root as the resampling
  unit.

With five roots, confidence intervals are descriptive and may be wide. Claims
must emphasize effect magnitude, consistency and raw outcomes rather than use a
binary significance label as proof.

Paired comparisons use within-root differences wherever roots are shared:

- actor sizes within an algorithm;
- algorithms at a given actor size; and
- Frenet versus LiDAR in Experiment 2.

Confidence intervals for paired effects resample root-level differences, not
individual episodes or checkpoints. Experiment 2 also reports the distribution
across test tracks, but the primary observation effect is the paired difference
between each root's track-aggregated score.

Learning curves are aligned on training interactions. Curves report a central
estimate with a seed-variation band and retain individual seed traces in a
supplementary plot. Smoothing, if used for presentation, never changes the raw
values used for convergence or area calculations and its window is stated.

Operational failures are distinguished from learning failures:

- an invalid artifact, exception or hardware interruption is repaired and the
  same run specification is rerun;
- a valid run that crashes, performs poorly or never converges is a scientific
  outcome and remains in the analysis; and
- convergence times are not averaged only over successful runs without also
  reporting success fraction and censoring.

## Required Tables and Figures

### Experiment 1

- configuration and actor/critic parameter-count table;
- final task metrics for all nine cells with raw seed points;
- completion/crash counts by algorithm and actor size;
- evaluation return and progress learning curves against interactions;
- interactions and wall time to convergence with censored runs marked;
- normalized learning-curve area by actor size and algorithm;
- final performance versus actor parameter count;
- throughput, peak memory and end-to-end runtime versus actor size;
- between-seed variability by cell;
- optimization diagnostic panels for each algorithm; and
- curvature-conditioned speed/throttle plot for representative final policies.

### Experiment 2

- selected PPO-size calculation;
- split manifest and track-geometry summary;
- held-out completion, crash, progress, return and lap-time comparison;
- paired root-level Frenet-minus-LiDAR differences;
- per-track paired performance distribution;
- training-reference, validation and test generalization gaps;
- validation learning curves against interactions;
- performance stratified by predeclared track-geometry bins; and
- computation, memory and optimization-diagnostic comparison.

Every table and figure is generated from raw artifacts by the repository
analysis command. Manually edited summary numbers are not authoritative.

## Artifact Layout

The exact root path is locked during Phase 2. Conceptually, each run contains:

```text
results/
  <run-kind>/
    <experiment>/
      <run-id>/
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

Aggregated tables and figures live outside individual run directories and record
the input run IDs and checksums. Large routine checkpoints and raw result trees
are not committed by default. Compact manifests, protocol revisions, smoke
fixtures and final summary tables needed to audit claims may be committed when
approved.

## Interpretation Rules

- Network-size claims are made within each algorithm first. The policy-size
  factor does not alter critic size.
- Algorithm claims are secondary and acknowledge different estimators and
  calibrated update rules.
- Completion and crash counts accompany conditional lap-time statistics.
- Sample efficiency uses environment interactions; computational efficiency
  uses measured time and throughput on the fixed machine.
- A non-converged run is not assigned the mean convergence time of successful
  runs.
- The Experiment 2 observation comparison is paired by training root and track
  schedule.
- A small generalization gap is meaningful only with adequate absolute test
  performance.
- PPO actor size is selected before viewing Experiment 2 test outcomes.
- Negative results, high variance and violations of the expected algorithm
  ordering are reported rather than tuned away.

## Known Limitations

- Five training roots provide only a modest estimate of stochastic variability.
- Experiment 1 uses one circuit, so its policy-size result is specific to that
  circuit and observation representation.
- The three actor sizes vary width while holding depth at two hidden layers;
  conclusions concern capacity at those chosen architectures, not every notion
  of neural-network complexity.
- Fixed critic capacity isolates actor size but may constrain the largest actor
  in actor-critic algorithms; critic diagnostics must be inspected.
- The secondary algorithm comparison cannot attribute differences solely to
  abstract “complexity,” because the estimators and update schedules differ.
- Experiment 2 tests only circuits from the frozen procedural generator
  distribution, not arbitrary real racing circuits.
- Tracks are assigned at episode resets. Policies with different episode
  lengths follow the same ordered seed schedule but can encounter different
  numbers of circuits within the common interaction budget; circuit exposure is
  therefore recorded and included when interpreting the observation comparison.
- Feed-forward LiDAR is intentionally partially observable and may be
  disadvantaged relative to a recurrent sensor policy.
- Wall-clock results apply to the recorded machine and software environment.
- The point-car and any minimal grip model remain simplified vehicle dynamics.

These limitations bound the claims; they do not justify changing the frozen
protocol after seeing measurement results.
