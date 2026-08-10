# Learning Contract

This document translates the policy-gradient algorithms in the course notes
into an exact implementation contract for the racing problem. It explains the
extra choices needed for bounded continuous actions instead of treating them as
if they came from the theory.

The main course references are
[`policy-gradient-1.md`](theory/policy-gradient-1.md),
[`policy-gradient-2.md`](theory/policy-gradient-2.md),
[`actor-critic.md`](theory/actor-critic.md) and
[`deep-rl.md`](theory/deep-rl.md).

## Configuration, seed and execution contract

The implementation stores immutable training and experiment configurations as
plain JSON-compatible dictionaries. Actor sizes are named `small`, `medium` and
`large`, but each serialized configuration retains its literal hidden-width pair:
`[32, 32]`, `[64, 64]` or `[256, 256]`. The critic always serializes as
`[64, 64]`. The learning rates remain deliberately unresolved until the
pre-experiment rule in [`EXPERIMENT.md`](EXPERIMENT.md) selects from its finite
candidates; no agent configuration supplies a silent final rate.

For a run with namespace $n$ and local identity $i$, all seed derivation starts
from `SeedSequence([20260810, n, i])`. Seven immutable child identities keep
independent randomness for actor initialization, critic initialization, policy
actions, environment resets, training-track scheduling, minibatch permutations,
and evaluation/reference work, in that order at child indices `0..6`. A child is
derived from its fixed index rather than by mutating a shared generator, so using
evaluation cannot change a training stream. Track-only identities use the same
protocol with their documented additional coordinates.

Seed derivation returns fresh NumPy generators or deterministic integer seeds
for PyTorch and environments without changing global NumPy or PyTorch RNG state.
At run startup, a separate explicit operation configures PyTorch for one
intra-op and one inter-op thread, deterministic algorithms with errors, and
disabled cuDNN benchmarking. That operation intentionally changes process-wide
PyTorch settings and must run before parallel PyTorch work. It cannot guarantee
bitwise equality across PyTorch versions, platforms, CPU versus GPU, or kernels
without deterministic implementations.

## Problem-specific notation

At agent timestep $t$, the policy receives an observation $O_t$. Depending on
the experiment, this is either

$$
O_t^{\mathrm{Frenet}}=(d_t,\phi_{e,t},v_t,\bar\kappa_t)
$$

or

$$
O_t^{\mathrm{LiDAR}}=(v_t,\widetilde r_t^{(1)},\ldots,
\widetilde r_t^{(16)}).
$$

It chooses the bounded action

$$
A_t=(A_t^{\mathrm{throttle}},A_t^{\mathrm{steer}})\in[-1,1]^2.
$$

After executing $A_t$, the environment returns reward $R_{t+1}$, next
observation $O_{t+1}$ and two booleans:

- `terminated` is true after completing the lap or crashing;
- `truncated` is true after reaching the $5000$-step time limit.

The actor has parameters $\mathbf\theta$. A2C and PPO additionally use a critic
$v_{\mathbf w}(O_t)$ with parameters $\mathbf w$. The training objective is

$$
J(\mathbf\theta)=
\mathbb E_{\pi_{\mathbf\theta}}
\left[\sum_{t=0}^{\tau}\gamma^tR_{t+1}\right],
\qquad \gamma=0.9995.
$$

The value of $\gamma$ comes from the task-timescale calculation in
[`MDP.md`](MDP.md): its effective horizon is $2000$ agent steps, or $80$ simulated
seconds. The undiscounted episode return is still recorded as a task metric;
$\gamma$ only controls learning targets.

## From a Gaussian policy to bounded controls

The course notes define a stochastic policy
$\pi_{\mathbf\theta}(\cdot\mid O_t)$ but deliberately do not prescribe a
distribution for continuous actions. This project makes the following design
choice.

The actor network produces two real numbers

$$
\boldsymbol\mu_{\mathbf\theta}(O_t)=
(\mu_t^{\mathrm{throttle}},\mu_t^{\mathrm{steer}}),
$$

which are the mean of an ordinary Gaussian before action bounds are applied.
The policy also learns two state-independent log standard deviations

$$
\boldsymbol\ell=(\ell^{\mathrm{throttle}},\ell^{\mathrm{steer}}),
\qquad
\boldsymbol\sigma=\exp(\boldsymbol\ell).
$$

Using a logarithm lets the optimizer change dispersion while guaranteeing
$\sigma>0$. A diagonal Gaussian means that the two exploratory noise samples
are independent once $O_t$ is fixed. Both means still depend on the complete
observation through the same neural network.

The Gaussian sample $U_t$ is unbounded, so it cannot be sent directly to the
environment. Sampling therefore has two steps:

$$
U_t=\boldsymbol\mu_{\mathbf\theta}(O_t)
+\boldsymbol\sigma\odot\varepsilon_t,
\qquad \varepsilon_t\sim\mathcal N(\mathbf0,I),
$$

where $\odot$ is the componentwise, or Hadamard, product. Thus
$U_{t,j}=\mu_{\mathbf\theta,j}(O_t)+\sigma_j\varepsilon_{t,j}$ for each action
component $j$.

This is the **reparametrization trick**: instead of viewing $U_t$ as an opaque
random draw whose distribution depends on the policy parameters, draw
$\varepsilon_t$ from a fixed standard Gaussian independent of those parameters,
then express $U_t$ as a deterministic differentiable function of
$\boldsymbol\mu_{\mathbf\theta}$, $\boldsymbol\sigma$ and $\varepsilon_t$. It
separates the source of randomness from the learned transformation. The present
algorithms still use the score-function gradient, so the sampled $U_t$ is
detached before the loss is formed; they do not use the pathwise gradient that
this parametrization could otherwise provide.

$$
A_t=\tanh(U_t).
$$

The hyperbolic tangent smoothly maps every real value into $(-1,1)$. Large
positive values approach full throttle or full left steering; large negative
values approach full braking or full right steering. This avoids a hard clip,
which would collapse many different Gaussian samples onto exactly the same
boundary action.

For deterministic evaluation, exploration is removed before the transformation:

$$
A_t^{\mathrm{eval}}=\tanh
\left(\boldsymbol\mu_{\mathbf\theta}(O_t)\right).
$$

### Why the log-probability needs a correction

REINFORCE, A2C and PPO all need
$\log\pi_{\mathbf\theta}(A_t\mid O_t)$. The Gaussian gives the probability
density of $U_t$, not of the squashed action $A_t$. A change of variables asks
how that density changes when the coordinates change from $U_t$ to
$A_t=\tanh(U_t)$. A small region around $U_t$ can shrink after `tanh`, especially
near the action bounds, so the same probability mass occupies a different
volume. The density must therefore be divided by this local volume change:

$$
\pi_{\mathbf\theta}(a\mid o)=
p_U(u\mid o)
\left|\det\frac{\partial\tanh(u)}{\partial u}\right|^{-1},
\qquad u=\operatorname{atanh}(a).
$$

Because `tanh` acts independently on the two components, its Jacobian is
diagonal with entries $1-\tanh^2(U_{t,j})$. Taking the logarithm turns the
determinant product into the sum used by every actor loss:

$$
\log\pi_{\mathbf\theta}(A_t\mid O_t)=
\sum_{j=1}^{2}
\left[
\log\mathcal N(U_{t,j};\mu_{\mathbf\theta,j}(O_t),\sigma_j^2)
-\log(1-\tanh^2(U_{t,j}))
\right].
$$

The sum makes this the probability of the complete throttle-steering vector,
not two unrelated loss terms.

Sampling retains $U_t$. If only $A_t$ is available, its components are first
clamped to $[-1+10^{-6},1-10^{-6}]$ before computing
$U_t=\operatorname{atanh}(A_t)$. The $10^{-6}$ is solely a numerical guard
against applying `atanh` to an exactly saturated floating-point action.

The sampled $U_t$ and $A_t$ are treated as constants inside the score-function
loss. Gradients flow through the log-probability, not backward through the
random sampling operation itself. Environment collection does not retain an
autograd graph. REINFORCE and A2C recompute current log-probabilities from stored
normalized observations and $U_t$ when forming their losses; PPO stores a
detached behaviour-policy log-probability and separately recomputes the current
one during optimization.

## Actor and critic architectures

Both models are ordinary fully connected multilayer perceptrons with two hidden
layers and `Tanh` activations.

For observation dimension $d_O$ and actor hidden widths $(h_1,h_2)$:

```text
normalized O_t in R^(d_O)
    -> Linear(d_O, h_1) -> Tanh
    -> Linear(h_1, h_2) -> Tanh
    -> Linear(h_2, 2)   -> mu_theta(O_t)
                                      + learned log-standard-deviation vector in R^2
                                      -> Gaussian sample U_t
                                      -> Tanh -> bounded action A_t
```

Experiment 1 changes only $(h_1,h_2)$:

- small actor: `(32, 32)`;
- medium actor: `(64, 64)`;
- large actor: `(256, 256)`.

Including the learned two-component log standard deviation, the actor parameter
count is

$$
(d_O+1)h_1+(h_1+1)h_2+(h_2+1)\cdot2+2.
$$

The critic architecture is fixed for every A2C and PPO comparison:

```text
normalized O_t in R^(d_O)
    -> Linear(d_O, 64) -> Tanh
    -> Linear(64, 64)  -> Tanh
    -> Linear(64, 1)   -> v_w(O_t)
```

Its parameter count is

$$
(d_O+1)\cdot64+(64+1)\cdot64+(64+1).
$$

Fixing the critic at `(64, 64)` prevents critic capacity from changing with the
actor-size factor. In the LiDAR comparison the input dimension necessarily
changes from $4$ to $17$, so the resulting input-layer parameter difference is
reported.

Hidden weights use orthogonal initialization with gain $\sqrt2$ and zero bias.
The actor mean output uses gain $0.01$, making initial mean actions close to
neutral; the critic output uses gain $1$. These gains are project engineering
choices, not consequences of the policy-gradient theorem.

The learned log standard deviations start at $-0.5$, corresponding to
$\sigma\approx0.61$ before squashing. This is a deliberately moderate initial
exploration scale. Values are constrained to $[-5,2]$ during use to avoid
numerical collapse or explosion. These three dispersion values are also project
choices and are checked during the pre-experiment configuration work.

## Input normalization and optimization safeguards

### Observation normalization

The observation components have incompatible scales: heading is measured in
radians, lateral displacement in metres, speed can approach $70$, and curvature
is much smaller. Feeding those raw magnitudes into one MLP would let scale alone
dominate early gradients.

For simplicity, each component uses naive two-pass-style running sums. After
$n$ observations, store the sum and squared sum

$$
S_n=\sum_{k=1}^{n}x_k,
\qquad
Q_n=\sum_{k=1}^{n}x_k^2.
$$

On a new value $x$, update $n\leftarrow n+1$, $S_n\leftarrow S_{n-1}+x$ and
$Q_n\leftarrow Q_{n-1}+x^2$. The population moments are then

$$
\mu_n=\frac{S_n}{n},
\qquad
\sigma_n^2=\max\left(\frac{Q_n}{n}-\mu_n^2,0\right).
$$

The maximum only removes a tiny negative value that floating-point cancellation
can create. Counts and sums are accumulated in 64-bit precision. A raw component
$x$ becomes

$$
\widetilde x=
\operatorname{clip}\left(
\frac{x-\mu_n}
{\sqrt{\sigma_n^2+10^{-8}}},
-10,10
\right).
$$

The $10^{-8}$ prevents division by zero before enough observations have been
seen. The range $[-10,10]$ is a project safeguard against a rare observation
producing an extreme network input; it does not alter the environment state or
reward.

Each current training observation updates the statistics before normalization.
A next observation used only to bootstrap a critic is normalized with the
current statistics but does not update them. Rollouts retain the normalized
values actually given to the networks. Evaluation freezes the statistics, so
evaluating more often cannot change training.

### Optimizer and gradient-norm clipping

Actor and critic use separate Adam optimizers. The moment parameters
$\beta_1=0.9$, $\beta_2=0.999$ and numerical constant $10^{-8}$ are the defaults
recommended in the original Adam paper. Learning rates are selected separately
during pre-experiment configuration because the course equations do not
determine them.

After backpropagation and before `optimizer.step()`, compute the global Euclidean
norm of all gradients belonging to one network. If the norm exceeds $0.5$, scale
that network's gradients together so their norm becomes $0.5$. Actor and critic
are clipped separately, and the norm before clipping is logged. This is a
project stability safeguard: one unusually noisy batch cannot produce an
arbitrarily large parameter update. It rescales the gradient vector without
changing the loss equation, clipping individual parameters or clipping actions.

This is distinct from **PPO ratio clipping**. Gradient-norm clipping is applied
to every algorithm after backpropagation and uses the norm threshold $0.5$.
PPO ratio clipping appears only inside the PPO actor objective: it limits the
importance ratio $\omega_t$ to $[1-\epsilon,1+\epsilon]$ in one surrogate branch,
with $\epsilon=0.2$, so a sample cannot keep rewarding a policy change that has
moved too far from the behaviour policy. It changes PPO's objective but does not
directly clip its gradients. PPO value clipping would be a third, separate
mechanism; it is disabled here.

No entropy bonus, weight decay, learning-rate scheduler, PPO value clipping or
KL early stop is enabled. The sampled value
$-\log\pi_{\mathbf\theta}(A_t\mid O_t)$ is logged only as a dispersion diagnostic.

## Episode endings and value bootstrap

The environment distinguishes genuine MDP endings from an external time limit.
The equations use explicit cases rather than multiplying by numeric masks.

For A2C and PPO, define the next-state bootstrap value as

$$
B_t=
\begin{cases}
0,
& \text{if `terminated` is true},\\
v_{\mathbf w}(O_{t+1}),
& \text{otherwise}.
\end{cases}
$$

A finish or crash therefore has no future value. A time-limit ending is not an
MDP terminal state, so it still uses the critic's estimate of $O_{t+1}$. The TD
error is

$$
\delta_t^{\mathbf w}=
R_{t+1}+\gamma B_t-v_{\mathbf w}(O_t).
$$

GAE is computed backward as

$$
\widehat{\mathbb A}_t=
\begin{cases}
\delta_t^{\mathbf w},
& \begin{aligned}
  &\text{if `terminated` or `truncated` is true, or}\\
  &\text{if this is the final transition stored in the rollout},
  \end{aligned}\\[6pt]
\delta_t^{\mathbf w}+\gamma\lambda\widehat{\mathbb A}_{t+1},
& \text{otherwise}.
\end{cases}
$$

Thus a time-limit transition bootstraps once through $B_t$ but does not continue
the recursion into the reset episode. A rollout cut behaves similarly: its TD
error bootstraps, while GAE does not refer to data outside the collected batch.
The fixed value target is

$$
y_t=\operatorname{detach}\left(
\widehat{\mathbb A}_t+v_{\mathbf w}(O_t)
\right).
$$

$\operatorname{detach}(x)$ has the same numerical value as $x$ but no autograd
path through the expression that produced it. This is the mathematical notation
used here for PyTorch's `Tensor.detach()` operation.

The raw advantage creates $y_t$. For the actor only, advantages are standardized
once over the rollout. PPO retains those same standardized values for all
optimization epochs.

## REINFORCE

### Purpose and expected behaviour

REINFORCE is the simplest algorithm in the comparison. It directly applies the
score-function estimator from the policy-gradient notes and has no critic. It
therefore provides the cleanest baseline for studying actor capacity, but its
Monte Carlo targets can have high variance and it cannot learn from an episode
until that episode ends.

Because there is no value function from which to bootstrap, REINFORCE collects
complete episodes. Eight episodes form one update batch so the update averages
several independently generated trajectories. The choice of eight is a project
trade-off: fewer episodes update more frequently but noisily, while more delay
every update and require more memory.

For complete trajectory $i$, compute return-to-go backward:

$$
G_t^i=
\begin{cases}
R_{t+1}^i,
& \text{if `terminated` or `truncated` is true},\\
R_{t+1}^i+\gamma G_{t+1}^i,
& \text{otherwise}.
\end{cases}
$$

Across every transition in the eight-episode batch, standardize the returns as
$\widetilde G_t^i=(G_t^i-\overline G)/(s_G+10^{-8})$. This supplies a non-learned
batch baseline and scale, following the variance-reduction motivation in the
course notes. It does not add a critic.

The actor minimizes

$$
\mathcal L_{\mathrm{REINFORCE}}(\mathbf\theta)=
-\frac1n\sum_{i=1}^{n}\sum_{t=0}^{\tau_i}
\log\pi_{\mathbf\theta}(A_t^i\mid O_t^i)
\operatorname{detach}(\widetilde G_t^i),
\qquad n=8.
$$

### REINFORCE pseudocode

```text
Input:
    actor architecture and actor learning rate
    training-interaction budget
    independent RNGs for actor initialization, policy sampling and reset

Initialize:
    actor parameters theta and learned log standard deviations
    observation running statistics
    actor Adam optimizer
    total_training_interactions <- 0

While enough budget remains to continue collecting:
    trajectories <- empty list

    For trajectory_index = 1, ..., 8:
        reset the fixed-track environment at the canonical start
        receive O_0
        trajectory <- empty list
        terminated <- false
        truncated <- false

        While neither terminated nor truncated and budget remains:
            update observation statistics with O_t
            normalize O_t to obtain the exact actor input
            compute mu_theta(O_t) and the current standard deviations
            sample pre-squash U_t with the policy RNG
            set A_t <- tanh(U_t)
            execute A_t and receive R_(t+1), O_(t+1), terminated, truncated
            store normalized O_t, U_t, A_t, reward and booleans
            increment total_training_interactions

        If the budget interrupted the episode before an environment boundary:
            retain the interactions and episode metrics for accounting
            do not use this incomplete trajectory in a Monte Carlo update
            stop collection
        Else:
            append the completed trajectory

    If fewer than 8 complete trajectories were collected before the budget:
        retain all interactions and episode records for accounting
        do not optimize from this incomplete batch
        stop training

    For every completed trajectory, moving backward from its last transition:
        set G_t <- R_(t+1) at termination or time-limit truncation
        otherwise set G_t <- R_(t+1) + gamma * G_(t+1)

    standardize all G_t values in the batch
    recompute corrected log pi_theta(A_t | O_t) from stored inputs and U_t
    compute each trajectory's sum of log-probability times detached G_t
    average those trajectory losses to obtain L_REINFORCE
    clear actor gradients
    backpropagate L_REINFORCE
    record the actor gradient norm and clip it if it exceeds 0.5
    apply one Adam update to theta and the log standard deviations
    log losses, dispersion, parameter norms, update size and interaction count

Save the final actor, normalizer, optimizer, RNG states and counters.
```

If the global budget ends during an episode or before eight new complete
episodes are available, all interactions and completed-episode records remain
counted but that incomplete update batch is not optimized. Recomputing
log-probabilities after collection avoids retaining a neural-network computation
graph for as many as 40,000 environment transitions. This keeps the reported
interaction budget exact without inventing a critic for REINFORCE.

## A2C with Generalized Advantage Estimation

### Purpose and difference from REINFORCE

A2C adds the value critic $v_{\mathbf w}$. The critic supplies bootstrap values,
so collection can stop after a fixed number of transitions instead of waiting
for eight episode endings. GAE combines successive TD errors to reduce variance,
at the cost of bias from the learned critic. The course notes describe this as a
synchronous batched V-critic actor-critic method.

One rollout contains $2048$ transitions and may cross several episode
boundaries. This is an engineering balance between update frequency and a less
noisy batch; unlike REINFORCE, it is not imposed by the mathematics. The GAE
parameter is $\lambda=0.95$. This is a conventional middle point between the
one-step case $\lambda=0$ and the higher-variance limit near $1$, and will be
checked before the reported experiment rather than presented as a theorem.

With $N$ transitions, A2C minimizes separate mean losses

$$
\mathcal L_{\mathrm{actor}}(\mathbf\theta)=
-\frac1N\sum_{t=0}^{N-1}
\log\pi_{\mathbf\theta}(A_t\mid O_t)
\operatorname{detach}(\widetilde{\mathbb A}_t),
$$

$$
\mathcal L_{\mathrm{critic}}(\mathbf w)=
\frac1{2N}\sum_{t=0}^{N-1}
\left(v_{\mathbf w}(O_t)-y_t\right)^2.
$$

The actor loss cannot update $\mathbf w$, and the critic target cannot backpropagate
through the values used to construct it.

### A2C+GAE pseudocode

```text
Input:
    actor and fixed critic architectures
    actor and critic learning rates
    gamma = 0.9995, lambda = 0.95, rollout capacity = 2048
    training-interaction budget and independent RNG streams

Initialize:
    actor parameters theta and learned log standard deviations
    critic parameters w
    observation running statistics
    separate actor and critic Adam optimizers
    total_training_interactions <- 0
    reset the environment and receive O_0

While total_training_interactions < budget:
    rollout <- empty list
    target_rollout_length <- min(2048, remaining interaction budget)

    For rollout_step = 1, ..., target_rollout_length:
        update observation statistics with current O_t
        normalize O_t for actor and critic
        compute mu_theta(O_t), standard deviations and v_w(O_t)
        sample U_t with the policy RNG and set A_t <- tanh(U_t)
        compute corrected log pi_theta(A_t | O_t)
        execute A_t and receive R_(t+1), O_(t+1), terminated, truncated

        if terminated:
            set bootstrap value B_t <- 0
        otherwise:
            normalize O_(t+1) without updating observation statistics
            set B_t <- v_w(O_(t+1))

        store the exact normalized inputs, U_t, action, detached value, reward,
            detached bootstrap value, booleans, episode identity and track identity
        increment total_training_interactions

        if terminated or truncated:
            reset the environment and use its observation as the next current O_t
        otherwise:
            set current O_t <- O_(t+1)

    Moving backward through the rollout:
        compute delta_t <- R_(t+1) + gamma * B_t - v_w(O_t)
        if terminated, truncated or final stored rollout transition:
            set raw advantage Ahat_t <- delta_t
        otherwise:
            set Ahat_t <- delta_t + gamma * lambda * Ahat_(t+1)
        set detached critic target y_t <- Ahat_t + v_w(O_t)

    standardize the raw advantages for the actor only
    recompute corrected log pi_theta(A_t | O_t) from stored inputs and U_t
    compute the mean actor loss over the entire rollout
    clear actor gradients, backpropagate, record norm, clip and update theta
    compute the mean half-squared critic loss over the entire rollout
    clear critic gradients, backpropagate, record norm, clip and update w
    log actor, critic, advantage, target and optimization diagnostics

Save actor, critic, normalizer, both optimizers, RNG states and counters.
```

## Proximal Policy Optimization

### Purpose and difference from A2C

PPO keeps the A2C critic and GAE targets but reuses each rollout for several
optimization epochs. Reuse makes the current policy differ from the policy that
collected the actions. PPO therefore applies the importance ratio and clipped
surrogate objective presented in the course notes, limiting overly optimistic
improvements while retaining the sampled data.

The original PPO paper defines this repeated-minibatch structure. Its
continuous-control configuration motivates the starting choices of $2048$
rollout transitions, $10$ epochs and clipping parameter $\epsilon=0.2$. This
project uses minibatches of $64$, matching that reference configuration rather
than the previously unexplained value $256$.

Store and detach the behaviour-policy log-probability at collection time. During
an update,

$$
\omega_{\mathbf\theta/\mathbf\theta_{\mathrm{old}},t}
=\exp\left(
\log\pi_{\mathbf\theta}(A_t\mid O_t)
-\log\pi_{\mathbf\theta_{\mathrm{old}}}(A_t\mid O_t)
\right).
$$

For minibatch $B$, minimize

$$
\mathcal L_{\mathrm{actor}}(\mathbf\theta)=
-\frac1{|B|}\sum_{t\in B}
\min\left\{
\omega_t\widetilde{\mathbb A}_t,
\operatorname{clip}(\omega_t,1-\epsilon,1+\epsilon)
\widetilde{\mathbb A}_t
\right\},
$$

$$
\mathcal L_{\mathrm{critic}}(\mathbf w)=
\frac1{2|B|}\sum_{t\in B}
\left(v_{\mathbf w}(O_t)-y_t\right)^2.
$$

Old log-probabilities, standardized advantages and value targets remain fixed
for all ten epochs. Each seeded epoch permutation covers every rollout row once.

### PPO pseudocode

```text
Input:
    actor and fixed critic architectures
    actor and critic learning rates
    gamma = 0.9995, lambda = 0.95
    rollout capacity = 2048, minibatch size = 64
    update epochs = 10, clipping epsilon = 0.2
    training-interaction budget and independent RNG streams

Initialize:
    actor parameters theta and learned log standard deviations
    critic parameters w
    observation running statistics
    separate actor and critic Adam optimizers
    total_training_interactions <- 0
    reset the environment and receive O_0

While total_training_interactions < budget:
    theta_old denotes the policy used for this collection
    rollout <- empty list
    target_rollout_length <- min(2048, remaining interaction budget)

    For rollout_step = 1, ..., target_rollout_length:
        update and apply observation normalization exactly as in A2C
        compute mu_theta_old(O_t), standard deviations and v_w(O_t)
        sample U_t, form bounded A_t and compute old_log_probability_t
        execute A_t and receive the environment transition
        set bootstrap value to zero after true termination
        otherwise evaluate v_w on the normalized next observation
        store normalized inputs, U_t, A_t, detached old log-probability,
            detached old value, reward, detached bootstrap value, booleans,
            episode identity and track identity
        increment total_training_interactions
        reset after termination or truncation; otherwise continue the episode

    Moving backward through the rollout:
        compute TD errors, raw GAE advantages and detached critic targets
        stop recursion at termination, truncation and the rollout boundary

    standardize advantages once and keep them fixed
    keep old log-probabilities and critic targets fixed

    For epoch = 1, ..., 10:
        create a permutation using the dedicated minibatch RNG
        divide the permutation into minibatches of 64 without replacement

        For each minibatch B:
            recompute log pi_theta(A_t | O_t) under the current actor
            compute omega_t from current minus old log-probability
            compute the clipped actor loss using fixed advantages
            clear actor gradients, backpropagate, record norm, clip and update theta

            compute the unclipped half-squared critic loss using fixed y_t
            clear critic gradients, backpropagate, record norm, clip and update w

            log actor loss, critic loss, approximate KL, clip fraction,
                importance-ratio statistics and gradient diagnostics

    verify that every rollout row appeared exactly once in every epoch
    discard the rollout before collecting with the updated policy

Save actor, critic, normalizer, both optimizers, RNG states and counters.
```

## Provenance of numerical choices

The following distinction is intentional:

| Choice | Origin |
|---|---|
| Policy-gradient, actor-critic, GAE and clipped-PPO equations | Repository course notes |
| $\gamma=0.9995$ | Racing timescale derivation in `MDP.md` |
| Actor widths | Scientific factor fixed by the experiment design |
| Fixed `(64, 64)` critic | Project control that prevents a critic-capacity confound |
| Adam $\beta_1$, $\beta_2$ and $10^{-8}$ | Defaults recommended in the original Adam paper |
| PPO rollout 2048, 10 epochs, minibatch 64 and clip 0.2 | Starting configuration reported for continuous control in the original PPO paper |
| $\lambda=0.95$ | Conventional GAE/PPO starting value, checked before reported runs |
| Squashed Gaussian, state-independent dispersion and its bounds | Explicit project policy-class choice required by the bounded action space |
| Naive running-sum normalization | Explicit project simplicity choice |
| Initialization gains, observation clipping and gradient norm 0.5 | Explicit project stability choices, not derived from course theory |
| REINFORCE batch of 8 and A2C rollout of 2048 | Explicit collection trade-offs, checked before reported runs |

Primary external references for choices absent from the course notes are the
[GAE paper](https://arxiv.org/abs/1506.02438),
[PPO paper](https://arxiv.org/abs/1707.06347),
[Adam paper](https://arxiv.org/abs/1412.6980) and
[PyTorch reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html).
