# Experimental Protocol

This document specifies the two reported experiments:

* **Experiment 1**. Measuring the effect of the actor-network size on the task of learning to race in one fixed circuit. 
* **Experiment 2**. Measuring the ability of PPO to generalize to multiple circuits, under different observations (Frenet vs LiDAR).

The exact policy, model, target and loss definitions for the training are reported in [`LEARNING.md`](LEARNING.md). 

For all comparisons, all algorithms are given a **fixed budget of interactions**, instead of episodes or number of updates.
This stresses how much each algorithm is capable of making treasure of each transition observed with the environment.

## Preliminary Steps

### 1. Reproducible Randomness Setup

One integer written on a run specification is called its **root identity**. 
It must control the whole run reproducibly, but using one mutable random generator for everything would create accidental coupling. 
For example, adding an evaluation could consume random numbers and change every later training action.

The solution is to derive independent child generators for distinct jobs:
1. actor initialization;
2. critic initialization;
3. stochastic policy actions;
4. environment reset;
5. training-circuit schedule;
6. PPO minibatch order; and
7. evaluation or baseline-policy sampling where sampling exists; and
8. track generation.

This hierarchy is implemented through `numpy.random.SeedSequence`.
Using `SeedSequence`, a single stream of random numbers is identified by the quadruple:
$$ [\mathtt{PROTOCOL\,KEY}, \ \mathtt{Namespace\,Code}, \ \mathtt{Local\,Identity}, \ \mathtt{Stream}] $$
Where:
* $\mathtt{PROTOCOL\,KEY}$ is always set to `0`.
* $\mathtt{Namespace\,Code}$ describes the experiment that is currently being run. 
  It assumes values from $1$ to $12$ according to the following table:
  | Purpose | Namespace code |
  |---|---:|
  | Experiment 1 reported roots | 1 | 
  | Experiment 2 reported roots | 2 | 
  | Learning-rate configuration | 3 | 
  | Capability check | 4 | 
  | Reduced-budget end-to-end validation | 5 |
  | Controlled-problem algorithm validation | 6 | 
  | Experiment 1 circuit candidates | 7 | 
  | Multi-circuit development checks | 8 | 
  | Experiment 2 training circuits | 9 |
  | Experiment 2 validation circuits | 10 | 
  | Experiment 2 test circuits | 11 | 
  | Randomized execution order | 12 |
* $\mathtt{Local\,Identity}$ is the **local identity** of a specific run. 
  It starts from $0$ and grows one by one for each repeated run.
* $\mathtt{Stream}$ represents one of the $8$ child generators, that distinguish the specific stream of random numbers being used. 
  Its values go from $1$ to $8$. 

Every recorded run stores both human-readable logical identities and generated integer states.
Checkpoints retain all mutable generator states required for an exact resume on the supported hardware and software stack.

### 2. Learning Rate Search

To determine a proper **learning rate** for all of the algorithms, different values are tested for each of them, in a short run using:
* An actor network of size `(64, 64)`.
* A budget of $250\,000$ interactions, which is increased to $750\,000$ if no learning rate prevails.
* $3$ seed roots.

The tested values are the following:

- REINFORCE Actor rate:
  $$1\cdot10^{-4}, \quad 3\cdot10^{-4}, \quad 1\cdot10^{-3}$$

- A2C/PPO Actor and Critic rates:
  $$ 
  (1\cdot10^{-4}, 3\cdot10^{-4}), \quad (3\cdot10^{-4}, 1\cdot10^{-3}), \quad (3\cdot10^{-4}, 3\cdot10^{-3}), \\
  (3\cdot10^{-4}, 1\cdot10^{-2}), \quad (1\cdot10^{-3}, 1\cdot10^{-3}), \quad (1\cdot10^{-3}, 3\cdot10^{-3}), \\
  (1\cdot10^{-3}, 1\cdot10^{-2})
  $$
  
The candidates are engineering scales around Adam's $1\cdot10^{-3}$ suggested
default. 

#### Selection Criteria

Selection of the **LR** is done using $4$ criteria, applied one after the other:
1. **Laps**. Number of final deterministic evaluations that complete a full lap.
2. **Mean progress**. If tied, compare mean maximum normalized progress achieved in the final deterministc evaluations.
3. **Mean return**. If still tied, compare mean return achieved in the final deterministic evaluations.
4. **Scale**. If still tied, choose by smaller actor LR, then smaller critic LR.  

#### Recorded Outcomes

| Algorithm | Actor | Critic | Allowance | Laps | Mean progress | Mean return |
|---|---:|---:|---:|---:|---:|---:|
| REINFORCE | $1\cdot10^{-4}$ | — | 250k | 0/3 | 0.070 | −4.28 |
| REINFORCE | $3\cdot10^{-4}$ | — | 250k | 0/3 | 0.169 | 4.77 |
| **REINFORCE** | $\mathbf{1\cdot10^{-3}}$ | — | 250k | **2/3** | **0.894** | **152.67** |
| A2C | $1\cdot10^{-4}$ | $3\cdot10^{-4}$ | 750k | 0/3 | 0.265 | 9.54 |
| A2C | $3\cdot10^{-4}$ | $1\cdot10^{-3}$ | 750k | 0/3 | 0.274 | 14.10 |
| A2C | $3\cdot10^{-4}$ | $3\cdot10^{-3}$ | 750k | 0/3 | 0.520 | 32.77 |
| A2C | $3\cdot10^{-4}$ | $1\cdot10^{-2}$ | 750k | 0/3 | 0.582 | 37.23 |
| A2C | $1\cdot10^{-3}$ | $1\cdot10^{-3}$ | 750k | 2/3 | 0.835 | 150.51 |
| **A2C** | $\mathbf{10^{-3}}$ | $\mathbf{3\cdot10^{-3}}$ | 750k | **3/3** | **1.000** | **222.77** |
| A2C | $1\cdot10^{-3}$ | $1\cdot10^{-2}$ | 750k | 3/3 | 1.000 | 218.50 |
| PPO | $1\cdot10^{-4}$ | $3\cdot10^{-4}$ | 250k | 3/3 | 1.000 | 225.52 |
| PPO | $3\cdot10^{-4}$ | $1\cdot10^{-3}$ | 250k | 3/3 | 1.000 | 223.77 |
| PPO | $3\cdot10^{-4}$ | $3\cdot10^{-3}$ | 250k | 3/3 | 1.000 | 227.19 |
| **PPO** | $\mathbf{3\cdot10^{-4}}$ | $\mathbf{10^{-2}}$ | 250k | **3/3** | **1.000** | **231.00** |
| PPO | $1\cdot10^{-3}$ | $1\cdot10^{-3}$ | 250k | 3/3 | 1.000 | 207.57 |
| PPO | $1\cdot10^{-3}$ | $3\cdot10^{-3}$ | 250k | 3/3 | 1.000 | 225.83 |
| PPO | $1\cdot10^{-3}$ | $1\cdot10^{-2}$ | 250k | 2/3 | 0.686 | 148.50 |

**Selected learning rates** therefore are: 

| Algorithm | Actor | Critic |
|---|---:|---:|
| REINFORCE | $1\cdot10^{-3}$ | — |
| A2C | $1\cdot10^{-3}$ | $3\cdot10^{-3}$ |
| PPO | $3\cdot10^{-4}$ | $1\cdot10^{-2}$ |

### 3. Fixed Circuit Choice for Experiment 1

The fixed circuit was chosen by manually filtering interesting generated circuits among the first $20$ seeds.

The selected circuit was the one with `seed=0`.

### 4. Hardware Choice and Timing

Everything runs on the CPU with PyTorch `float32`. 
Neural work is not placed on a GPU, and no GPU path is offered.

The actor and critic have two hidden layers and are stepped in batches of one row per environment worker, which is far too small to amortize host-to-device transfer. 
Over 20,000 interactions with $8$ workers, these were the measured speeds:

| Algorithm | CPU | CUDA |
|---|---:|---:|
| REINFORCE | 3,372 interactions/s | 1,862 interactions/s |
| A2C | 3,694 interactions/s | 1,144 interactions/s |
| PPO | 2,835 interactions/s | 1,146 interactions/s |

All three algorithms use the same worker count ($8$, the number of physical CPU cores), so that reported collection throughput and wall time are comparable. 

Timing categories, measured during the runs, are selected so that they do not overlap:

- Environment collection time;
- Actor/critic optimization time;
- Deterministic evaluation time;
- Checkpoint and metric-persistence time;
- End-to-end time.

Reported runs execute one at a time; no other training process competes for the cores.


## Experiment 1 — Actor size on one circuit

### Research Question

> **RQ1**: How does actor-network capacity affect training an agent to race a car on one fixed circuit?

The actor-size comparison is repeated for REINFORCE, A2C+GAE and PPO. 
The algorithm comparison is secondary: it describes the practical effect of adding a critic, GAE and bounded sample reuse.

**RQ1** is answered by observing, in practice, four measures of training efficiency and efficacy:
* **Final driving performance**. How fast does the agent complete the lap (if they manage to complete it)?
* **Interaction efficiency**. How many interactions with the environment does the agent have to perform before reaching the task threshold?
* **Converge reliability**. How do different runs under the same settings vary in the results?
* **Computational cost**. How long does the training take for the agent to reach the task threshold?

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

* the saved `tracks/experiment_1.json` circuit and canonical start;
* Frenet observation $(d_t,\phi_{e,t},v_t,\delta_t,\bar\kappa_t)$;
* the same action mapping, reward, episode limit and frozen physics version;
* the bounded Gaussian policy and optimizer contract in `LEARNING.md`;
* one fixed `(64, 64)` critic for A2C and PPO;
* the learning rate selected before the experiment for its algorithm;
* 2,000,000 training interactions;
* deterministic evaluation every 50,000 training interactions; and
* checkpoints every 250,000 interactions and at the final budget.

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
the lap in at most $34$ simulated seconds. A run that never meets the rule is
right-censored at 2,000,000 interactions and remains in every success-rate and
learning-curve summary.

The threshold was previously twice the reference controller's average lap, which
was $32\,\mathrm s$ when that average was $15.9\,\mathrm s$. Doubling no longer
works: the reference averages $22.3\,\mathrm s$ on the current circuits and twice
that is $44.6\,\mathrm s$, above the $40\,\mathrm s$ episode cap, which would make
the time condition vacuous and silently reduce the rule to three consecutive
completed laps. The value is now about $1.5\times$ the reference average, between
its slowest lap and the cap. The multiple is a project choice, not a measurement.

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
- Python and dependency freeze, OS, CPU and PyTorch versions;
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
- gradient-estimator dispersion: the signal-to-noise ratio and mean pairwise
  cosine similarity of the actor gradient across equal disjoint sub-batches of
  the same update, with the sub-batch size and count;
- critic loss, gradient norm, weight norm, predictions, targets, advantages and
  explained variance for A2C and PPO;
- importance-ratio distribution, clip fraction and approximate KL for PPO; and
- interaction count and optimization duration.

The dispersion entry is the one measurement here that is not a by-product of
training. Estimator variance is what separates these algorithms in theory, and no
norm can show it: a norm describes one averaged gradient rather than how far an
equally sized second sample would have landed from it. The batch is therefore
split into disjoint sub-batches of a fixed 256 transitions, each estimating the
same gradient, and their spread is summarized scale-free so it compares across
algorithms whose losses carry different constant factors. The size is fixed
rather than proportional because estimator variance falls as one over the sample
count, so equal-sized samples are the only fair comparison.

It is measured before any optimizer step of the update. At that point PPO's
importance ratio is exactly one and its clipped surrogate reduces to the same
advantage-weighted estimator A2C uses, so the comparison is between *estimators*
rather than optimization schedules: it separates REINFORCE's return weighting
from the baselined weighting the other two share, and should not be expected to
separate A2C from PPO. Report both summaries. The cosine is the more reliable of
the two when sub-batch gradient magnitudes are heavy-tailed, as REINFORCE's are,
because the ratio is dominated by the largest sub-batch while the cosine
normalizes each one.

**Computational cost**

- collection, optimization, evaluation, persistence and end-to-end times;
- completed episodes and optimizer updates;
- interactions per collection second and per training second;
- actor, critic and total parameter counts; and
- peak process memory where available.

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

The splits are committed in `tracks/experiment_2_splits.json`, which records for
each circuit its identity, the generator seed that identity denotes, and its
length, straight fraction, curvature quantiles and tightest radius. The circuits
themselves are not stored: they are rebuilt from the frozen generator on demand,
and those statistics are re-checked on the way back in, so any change to the
generator is a loud failure rather than a silent change in what a circuit
identity means.

The comparison uses a relative tolerance of $10^{-6}$ rather than exact
equality, because a rebuild has to survive moving between machines. Generation
runs through `math.cos` and `math.sin`, whose last bit belongs to the platform's
maths library and not to the circuit, so a byte-for-byte comparison of
coordinates fails on Linux for circuits identical to within a picometre. A real
change to the generator moves this geometry by metres, and the recorded
statistics avoid quantities that sit near a decision boundary: every track
length is an exact multiple of the sample spacing, and no sampled curvature
comes within a factor of two of the straight/corner threshold.

The **training-reference** circuits are the exception to the split namespaces.
They are circuits the run itself trained on, taken in per-worker episode order
and revisited deterministically after training, so they carry the training
namespace by construction. They are deliberately in-sample: the gap between them
and the test circuits is what this experiment calls generalization. Because they
are selected by per-worker episode order, a paired root's two runs revisit the
same circuits.

There are **16** of them, matching the validation count. The training-reference
and validation summaries then rest on the same denominator, so their two gaps to
the test split can be compared with each other rather than with their own sample
sizes. Unlike the other splits they are not fixed across runs: each root trains
on its own circuits, so each root revisits its own sixteen.

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

Circuits change only at episode reset. Collection runs in parallel workers, and
each worker owns an identically seeded circuit-selection stream, so worker $w$
meets the same circuit on its $k$-th episode in both runs of paired root $i$.

Pairing is by *worker and per-worker episode count*, not by a global episode
index. Different observation policies produce episodes of different lengths, so
their episodes finish in a different order and a global index would name a
different circuit in each run — exactly the drift the pairing exists to prevent.
Per-worker counting is unaffected by when other workers happen to finish.

The two conditions therefore encounter the same circuits in the same per-worker
order, but not necessarily the same *number* of them within the common
interaction budget. Circuit exposure is recorded rather than silently treated as
identical.

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

The final policy is evaluated once on the 16 training-reference circuits, all 16
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
- throughput, parameter counts and peak process memory; and
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
  bins, whose edges are the tertiles of the eight development circuits and are
  recorded in the split commitment. They are chosen from circuits that exist to
  be looked at before the experiment, never from the results they describe. The
  curvature statistic is the 90th percentile of absolute sampled curvature: over
  half of every generated circuit is straight, so the median is exactly zero
  everywhere and separates nothing;
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
