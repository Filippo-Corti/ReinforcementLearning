# Learning Contract

This document describes the policy-gradient algorithms used for the project, together with the choices needed to adapt them specifically to the considered racing problem.

## Problem Specification

At agent timestep $t$, the policy $\pi_\theta$ receives an observation $O_t$. 
Depending on the experiment, the observation can be:

$$
O_t^{\mathrm{Frenet}}=(d_t,\phi_{e,t},v_t,\bar\kappa_t) \quad \text{or} 
\quad 
O_t^{\mathrm{LiDAR}}=(v_t,\widetilde r_t^{(1)},\ldots, \widetilde r_t^{(16)})
$$

The agent then chooses the bounded action:

$$
A_t=(A_t^{\mathrm{throttle}},A_t^{\mathrm{steer}}) \sim \pi_\theta(\cdot \mid O_t)
$$

After executing $A_t$, the environment returns reward $R_{t+1}$, next observation $O_{t+1}$ and the two booleans:
* `terminated` is true after completing the lap or crashing;
* `truncated` is true after reaching the configurable $T_{\max}$-step training time limit.

The actor, represented by the policy $\pi_\theta$, has parameters $\mathbf\theta$. 
A2C and PPO additionally use a critic, a $V$-function approximator $v_{\mathbf w}(O_t)$ with parameters $\mathbf w$. 

The training objective is the **performance function** for the policy:

$$
J(\mathbf\theta)=
\mathbb E_{\pi_{\mathbf\theta}}
\left[\sum_{t=0}^{\tau}\gamma^tR_{t+1}\right],
\qquad 
\gamma=0.9995.
$$

The value of $\gamma$ comes from the task-timescale calculation in
[`MDP.md`](MDP.md): its effective horizon is $2000$ agent steps, or $80$ simulated
seconds. The undiscounted episode return is still recorded as a task metric;
$\gamma$ only controls learning targets.
^ TODO: revisit if we decide to set $\gamma=1$ instead.

### From the Gaussian Policy to the Control Actions

The actor network produces two real numbers, which represent the mean of a bivariate Gaussian Distribution for the two action dimensions, throttle and steering:

$$
\boldsymbol\mu_{\mathbf\theta}(O_t)= (\mu_t^{\mathrm{throttle}},\mu_t^{\mathrm{steer}}),
$$

The policy also learns two state-independent log standard deviations:

$$
\boldsymbol\ell=(\ell^{\mathrm{throttle}},\ell^{\mathrm{steer}}),
\qquad
\boldsymbol\sigma=\exp(\boldsymbol\ell).
$$

Using a logarithm lets the optimizer move freely while guaranteeing $\sigma>0$.

The Gaussian sample $(\mu_\mathbf\theta, \ell)$ is unbounded, so it cannot be sent directly to the environment. 
Sampling therefore has two steps:

1. Sample an unbounded control action $U_t$:
    $$
    U_t=\boldsymbol\mu_{\mathbf\theta}(O_t) + \boldsymbol\sigma\odot\varepsilon_t,
    \qquad 
    \varepsilon_t\sim\mathcal N(\mathbf0,I),
    $$
    where $\odot$ is the componentwise product. 
2. Apply the bounding via hyperbolic tangent, which smoothly maps every value into $(-1, 1)$.
    $$ A_t = \tanh(U_t) $$

For deterministic evaluation, we can simply avoid adding the random noise:
$$ A_t^{\mathrm{eval}}=\tanh \left(\boldsymbol\mu_{\mathbf\theta}(O_t)\right) $$

### Log-Probability Computation

REINFORCE, A2C and PPO all need log-probabilities $\log\pi_{\mathbf\theta}(A_t\mid O_t)$. 
The Gaussian gives the probability density of $U_t$, not of the bounded action $A_t$. 
Notice that the $\tanh$ transformation alters the probability distribution, therefore:
$$ p_{\mathbf\theta}(U_t\mid O_t) \ne \pi_{\mathbf\theta}(A_t\mid O_t) $$

Using probability theory, we can derive $\pi_{\mathbf\theta}(A_t\mid O_t)$ given $p_{\mathbf\theta}(U_t\mid O_t)$ using:
$$
\pi_{\mathbf\theta}(A_t\mid O_t) = 
p_{\mathbf\theta}(\operatorname{atanh}(A_t) \mid O_t) 
\left|
\det \frac{\partial A_t}{\partial U_t}
\right|^{-1}
$$

From this, we derive the following log-probability computation:
$$
\log\pi_{\mathbf\theta}(A_t\mid O_t)=
\sum_{j=1}^{2}
\left[
\log\mathcal N(U_{t,j};\mu_{\mathbf\theta,j}(O_t),\sigma_j^2)
-\log(1-\tanh^2(U_{t,j}))
\right].
$$

Where the term $1-\tanh^2(U_{t,j})$ represents the derivative of $A_t = \tanh(U_t)$, with respect to $U_t$.

## Actor and Critic Architectures

Both models are ordinary fully connected multilayer perceptrons with two hidden
layers and `Tanh` activations.

For observation dimension $d_O$ and actor hidden widths $(h_1,h_2)$:
```
O_t ∈ ℝ^{d_O} 
    -> Linear(d_O, h_1) -> Tanh 
    -> Linear(h_1, h_2) -> Tanh 
    -> Linear(h_2 ,2) -> μ ∈ ℝ²
```

Experiment 1 changes $(h_1,h_2)$:
- Small actor: `(32, 32)`;
- Medium actor: `(64, 64)`;
- Large actor: `(256, 256)`.

Including the learned two-component log standard deviation, the actor parameter
count is:
$$ (d_O+1)h_1+(h_1+1)h_2+(h_2+1)\cdot2+2 $$

The critic architecture is fixed:
```
O_t ∈ ℝ^{d_O} 
    -> Linear(d_O, 64) -> Tanh 
    -> Linear(64, 64) -> Tanh 
    -> Linear(64, 1) -> v ∈ ℝ
```

Its parameter count is:
$$ (d_O+1)\cdot64+(64+1)\cdot64+(64+1). $$


| Model          | Hidden widths | Parameters ($d_{O_{\text{Frenet}}} = 5$) | Parameters ($d_{O_{\text{LiDAR}}} = 18$) |
| -------------- | ------------: | ---------------------: | ----------------------: |
| Actor — Small  |      (32, 32) |                  1,314 |                   1,730 |
| Actor — Medium |      (64, 64) |                  4,738 |                   5,570 |
| Actor — Large  |    (256, 256) |                 67,586 |                  70,914 |
| Critic         |      (64, 64) |                  4,609 |                   5,441 |

Hidden network weights are initialized using *orthogonal initialization*, with a gain (multiplier) equal to $\sqrt2$ and bias equal to $0$.
The learned log standard deviations start at $-0.5$, corresponding to
$\sigma\approx0.61$, and is bounded to $[-5, 0]$. 

## Input Normalization and Optimization Safeguards

### Observation Normalization

Since the observation components have incompatible scales, feeding their raw magnitudes into the MLPs would let scale dominate early gradients.

This issue is solved through normalization via **standardization with the previous measurements**:
* At each new observed value $x$, we update a running sum and running squared sum:
    $$
    S_n=\sum_{k=1}^{n}x_k, \qquad S_n\leftarrow S_{n-1}+x \\
    Q_n=\sum_{k=1}^{n}x_k^2, \qquad Q_n\leftarrow Q_{n-1}+x^2
    $$
* Then we can standardize $x$ via:
$$
\mu_n=\frac{S_n}{n}, \qquad \sigma_n^2=\max\left(\frac{Q_n}{n}-\mu_n^2,0\right) \\
\widetilde x = \frac{x-\mu_n}{\sqrt{\sigma_n^2}},
$$

### Optimizer and gradient-norm clipping

Actor and critic use separate Adam optimizers. 
The moment parameters $\beta_1=0.9$, $\beta_2=0.999$ and numerical constant $10^{-8}$ are the defaults recommended in the original Adam paper. 
Learning rates are selected separately during pre-experiment configuration.

After `loss.backward()` and before `optimizer.step()`, gradients are clipped by their global Elucidean norm:
$$ 
g \leftarrow g \cdot \min \left(1, \frac{0.5}{\Vert g \Vert_2} \right)
$$
This is a standard practice to improve stability, avoiding that unusually noisy batches produce arbitrarily large parameter updates.

## Episode Endings and Value Bootstrap

The environment distinguishes genuine MDP endings from an external time limit, using the `terminated` and `truncated` flags.

For A2C and PPO, the next-state bootstrap value is defined as as:
$$
B_t=
\begin{cases}
0, & \text{if terminated},\\
v_{\mathbf w}(O_{t+1}), & \text{otherwise}.
\end{cases}
$$

A time-limit ending is not an MDP terminal state, so it still uses the critic's estimate of $O_{t+1}$. The TD
error is:
$$
\delta_t^{\mathbf w}=
R_{t+1}+\gamma B_t-v_{\mathbf w}(O_t).
$$

The GAE Advantage is computed as:
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
G_t=\operatorname{detach}\left(
\widehat{\mathbb A}_t+v_{\mathbf w}(O_t)
\right).
$$

The raw advantage creates $G_t$. For the actor only, advantages are standardized
once over the rollout. PPO retains those same standardized values for all
optimization epochs.

# Learning Algorithms

All algorithms share the goal of maximizing the **performance function**:
$$ J(\theta) = \mathbb{E} \left[ \sum_{t=0}^{\tau}\gamma^t r(S_t, A_t) \right] $$

The performance function corresponds to the expected return from a full **episode** in which we apply the policy $\pi_\theta$.
The objective of the training is to find the parameters $\theta$ that describe the policy $\pi_\theta$ that has the maximum expected return over random episodes. 

All algorithms apply **stochastic gradient descent**, updating $\theta$ periodically:
$$ \theta_{k+1} \leftarrow \theta_k + \alpha \nabla J(\theta_k)  $$
Instead, they differ in the way they compute $J(\theta)$ and the gradient $\nabla J(\theta)$.

## REINFORCE

REINFORCE is the simplest algorithm considered in the experiment.
It is actor-only and Monte Carlo. 
This means that it:
1. Collects full episodes by interacting with the environment.
2. Uses the colected returns to build an estimate for $\nabla J(\theta)$.
3. Update $\theta$ using this estimate.

Its approach is:
* **High variance**, as playing the policy $\pi_theta$ for a full episode, with many steps, results in a long chain of choices which can easily cause large swings in the final return of the episode.
* **Low bias**, as it updates its estimate purely based on data it collects, without introducing any assumption or prior knowledge.

To limit the effect of variance on the updates, REINFORCE is implemented with **batches**, which for the experiment are set to be of size $n=8$.

### GPOMDP REINFORCE Estimator

For the implementation, we consider the **GPOMDP** variant of **REINFORCE**, which arranges the terms of the estimation in the REINFORCE procedure in a way that exploits the temporal structure of the MDP.

For a complete trajectory $i$, REINFORCE computes returns-to-go by iterating backwards on the recorded rewards:

$$
G_t^i=
\begin{cases}
r_{t+1}^i,
& \text{if `terminated` or `truncated` is true},\\
r_{t+1}^i+\gamma G_{t+1}^i,
& \text{otherwise}.
\end{cases}
$$

Then, it normalizes the returns-to-go using **standardization** as a **non-learned batch baseline**, which is a very simply variance-reduction technique:
$$ \widetilde G_t^i=\frac{G_t^i-\overline G}{s_G} $$

Finally, the gradient step is executed by using the **REINFORCE estimator**:
$$
\hat{\nabla} J(\theta) = 
\frac{1}{n} 
\sum_{i=1}^{n}\sum_{t=0}^{\tau_i} 
\log\pi_{\mathbf\theta}(A_t^i\mid O_t^i)(\widetilde G_t^i)
$$

> Note that, in the formulation, $G_\tau^i$ is the full return of the trajectory: standard REINFORCE uses this in its estimation. 
> Instead, GPOMDP REINFORCE uses all $G_t^i$, the **returns-to-go**.
> This distinction makes sure that probabilities are correcly aligned with the parts of the reward they contribute to, instead of assuming that all probabilities contribute to all of it.

### REINFORCE Pseudocode

```
Input:
    a: learning rate
    B: interaction budget (total number of allowed interactions)
    N: batch size (default N = 8)
    γ: discount term 

# Initialize variables
Adam <- torch.optim.Adam(a, ...)
interactions <- 0

while interactions < B:
    batch ← ∅
    # Collect trajectories for a full batch
    while |batch| < N and interactions < B:
        for each active environment:
            O_t <- normalize(O_t)        
            U_t ∼ 𝓝(μ_θ(O_t), σ²)      
            A_t <- tanh(U_t)
            O_t+1, R_t+1, done <- env.step(A_t)
            store(O_t, U_t, A_t, R_t+1)
            interactions <- interactions + 1
            if done:
                batch ← batch ∪ {τ}
    
    # Stop if budget does not allow enough trajectories to fill batch
    if |batch| < N: break

    # Compute returns-to-go and standardize them
    for each τ ∈ batch:
        G <- 0
        for t <- T-1,...,0:
            G_t <- R_t+1 + γG
    G <- standardize(G)

    # Compute REINFORCE loss (use stored data for the log-probs)
    L <- 0
    for τ ∈ batch:
        L <- L - (1/T) Σ_t log π_θ(A_t | O_t) · detach(G_t)
    L <- L / N

    # Perform the update step (with clipping)
    Adam.zero_grad()
    ∇L <- backprop(L)
    ∇L <- ∇L · min(1, 0.5 / ||∇L||₂) 
    Adam.step()
```
^ TODO: verify if we actually use the GPOMDP cause in the pseudocode we report standard estimator with just one sum instead of two.

Code references:
* [`REINFORCE Agent`](../src/agents/reinforce.py)
* [`REINFORCE Training Engine`](../src/training/engines/reinforce.py)

## A2C with GAE

A2C is the first considered actor-critic algorithm.
It trades some **variance**, which is particularly high in actor-only algorithms like REINFORCE, for some **bias**, which is introduced by the addition of a **value-function approximator**.
The approximator (the critic $v_{\mathbf w}$) is used to build **bootstrap values** that are used in the construction of a new estimator for the performance function gradient.

In this implementation, A2C is enhanced with Generalized Advantage Estimation (GAE), which is a more sophisticated way of building the **advantage estimator**, utilizing TD errors from multiple steps instead of just the one-step TD error.

More precisely, A2C+GAE explores multiple environments until $N=2048$ transitions of experience are collected.
These can go over multiple episodes; unlike with REINFORCE we do not need an episode to finish in order to actively use it for the updates.
Of course, different episodes are handled separately.

For any transition happened at timestep $t$ of an episode, we:
* First, compute the one-step TD errors:
$$
\delta_t^{\mathbf w} = r_{t+1} + \gamma B_t - v_{\mathbf w}(O_t) \\
\text{where} \quad 
B_t = \begin{cases} 
0, & \text{if episode ends here} \\
v_{\mathbf w}(O_{t+1}) & \text{otherwise}
\end{cases}
$$
* Then, compute the GAE advantage estimator as a weighted sum of the TD errors until the end of the episode:
$$
\hat{\mathbb{A}}_t^{\text{GAE}} = \sum_{k=0}^{K_t-1} (\gamma \lambda)^k \delta_{t+k}^{\mathbf w}
$$
* Finally, compute the return-to-go equivalent (using $\hat{\mathbb{A}}_t^{\text{GAE}}$ as a shortcut):
$$ G_t = v_{\mathbf w}(O_t) + \hat{\mathbb{A}}_t^{\text{GAE}} $$

Once all transitions are processed, we can use all $N=2048$ of them as a single update batch.
We compute and optimize the two separate losses:

$$
\mathcal L_{\mathrm{actor}}(\mathbf\theta)=
-\frac{1}{N} 
\sum_{t=0}^{N-1}
\log\pi_{\mathbf\theta}(A_t\mid O_t)(\hat{\mathbb A}_t^{\text{GAE}}),
$$

$$
\mathcal L_{\mathrm{critic}}(\mathbf w)=
\frac{1}{2N}
\sum_{t=0}^{N-1}
\left(v_{\mathbf w}(O_t)-G_t\right)^2.
$$


### A2C+GAE Pseudocode

```
Input:
    a_actor: actor learning rate
    a_critic: critic learning rate
    B: interaction budget (total number of allowed interactions)
    N: rollout capacity (default N = 2048)
    γ: discount term (default γ = 0.9995)
    λ: GAE parameter (default λ = 0.95)

# Initialize variables
Adam_actor <- torch.optim.Adam(a_actor, ...)
Adam_critic <- torch.optim.Adam(a_critic, ...)
interactions <- 0

while interactions < B:
    rollout <- ∅

    # Collect a rollout of (at most) N transitions
    while |rollout| <  min(N, B - interactions):
        for each active environment:
            O_t <- normalize(O_t)
            U_t ∼ 𝓝(μ_0(O_t), σ²)
            A_t = tanh(U_t)
            log π <- corrected_log_prob(A_t, U_t, μ_0(O_t), σ)
            O_t+1, R_t+1, done <- env.step(A_t)
            if done: 
                B_t <- 0
            else:
                B_t <- v_w(normalize(O_t+1))

            store(O_t, U_t, A_t, R_t+1, B_t, ...)
            interactions <- interactions + 1

    # Compute TD errors, GAE advantages and critic targets
    for each environment:
        for t <- last,...,0:
            δ_t <- R_t+1 + γB_t - v_w(O_t)
            if terminated or truncated or t is the final rollout transition:
                Ahat_t <- δ_t
            else:
                Ahat_t <- δ_t + γλAhat_t+1
            G_t <- Ahat_t + v_w(O_t)

    # Standardize advantages for the actor only
    Ahat <- standardize(Ahat)

    # Compute actor loss, then perform actor update (with clipping)
    L_actor <- mean(-log π · Ahat)
    Adam_actor.zero_grad()
    ∇L_actor <- backprop(L_actor)
    ∇L_actor <- ∇L_actor · min(1, 0.5 / ||∇L_actor||₂)
    Adam_actor.step()

    # Compute critic loss, then perform critic update (with clipping)
    L_critic <- mean(1/2 · (v_w(O) - G_t)²)
    Adam_critic.zero_grad()
    ∇L_critic <- backprop(L_critic)
    ∇L_critic <- ∇L_critic · min(1, 0.5 / ||∇L_critic||₂)
    Adam_critic.step()

```
^ TODO: why do we compute the logprobs in the collection loop here, but we do it separately at end in REINFORCE? But then again we do the standardization? I need to understand these steps better.

Code references:
* [`A2C Agent`](../src/agents/a2c.py)
* [`A2C Training Engine`](../src/training/engines/a2c.py)

## PPO

PPO is another actor-critic algorithm, which employs a **sample reuse** technique to gain more experience from the same number of interactions.

In order for previous transitions to stay valuable even after they have already been used, PPO makes sure that each optimization step does not drift the policy too far into one direction. 
With this caution, previous samples can be reused if their log-probability is adjusted to the probability of them being observed with the current policy.

In practice, PPO does not optimize the performance function $J(\theta)$ directly.
Instead, it optimizies a *surrogate* objective which measures the **performance difference** between the new policies and the previous one, assuming the two policies have the same state distribution:
$$ 
J(\theta') - J(\theta) = \mathbb{E}_{\substack{S\sim d_{\pi}\\ A\sim\pi(\cdot\mid S)}}
\left[
\frac{\pi'(A\mid S)}{\pi(A\mid S)}
\mathbb{A}^{\pi}(S,A)
\right]
$$

Where the fraction $\omega_{\theta' / \theta} =\frac{\pi'(A\mid S)}{\pi(A\mid S)}$ is the **importance ratio**.
Importantly, the above estimator is a good proxy for the policy improvement **only if** the new policies does not move too far from the policy that generated the transitions.
To ensure that this holds, PPO clips $\omega_{\theta' / \theta}$.

### PPO Optimization

Just like A2C, PPO starts by collecting $N=2048$ transitions.
Then, for each of these transitions it computes:
* The GAE advantage estimator, just like A2C+GAE:
    $$
    \hat{\mathbb{A}}_t^{\text{GAE}} = \sum_{k=0}^{K_t-1} (\gamma \lambda)^k \delta_{t+k}^{\mathbf w}
    $$
* The critic target, just like A2C+GAE:
    $$ G_t = v_{\mathbf w}(O_t) + \hat{\mathbb{A}}_t^{\text{GAE}} $$
* The log-probability of the transition (unlike A2C+GAE, which can compute it later since the policy won't change):
    $$ \log \pi_\theta(A_t \mid O_t) $$

Then, PPO uses all $N$ transitions for $K$ times (epochs), permutating them and splitting them into mini-batches.
In each mini-batch, it:
* Computes the importance ratio using:
    $$ \omega_t = \exp(\log \pi_{\theta'}(A_t \mid O_t) - \log \pi_{\theta}(A_t \mid O_t)) $$
* Computes the two losses and updates the parameters based on them, making sure that the actor loss properly clips the importance ratio:
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
    \left(v_{\mathbf w}(O_t)-G_t\right)^2.
    $$

### PPO Pseudocode

```
Input:
    a_actor: actor learning rate
    a_critic: critic learning rate
    B: interaction budget (total number of allowed interactions)
    N: rollout capacity (default N = 2048)
    M: minibatch size (default M = 64)
    K: number of optimization epochs (default K = 4)
    γ: discount term (default γ = 0.9995)
    λ: GAE parameter (default λ = 0.95)
    ε: PPO clipping parameter (default ε = 0.2)
    KL_target: target KL divergence (default KL_target = 0.02)

# Initialize variables
Adam_actor <- torch.optim.Adam(a_actor, ...)
Adam_critic <- torch.optim.Adam(a_critic, ...)
interactions <- 0

while interactions < B:
    rollout <- ∅
    θ_old <- θ

    # Collect a rollout of (at most) N transitions
    while |rollout| < min(N, B - interactions):
        for each active environment:
            O_t <- normalize(O_t)
            μ_old <- μ_θ_old(O_t)
            σ <- actor standard deviations
            U_t ∼ 𝓝(μ_old, σ²)
            A_t <- tanh(U_t)
            old_log_π_t <- corrected_log_prob(A_t, U_t, μ_old, σ)
            O_t+1, R_t+1, done <- env.step(A_t)
            if done:
                B_t <- 0
            else:
                B_t <- v_w(normalize(O_t+1))

            store(O_t, U_t, A_t, old_log_π_t, R_t+1, B_t, ...)
            interactions <- interactions + 1

    # Compute TD errors, GAE advantages and critic targets
    for each environment:
        for t <- last,...,0:
            δ_t <- R_t+1 + γB_t - v_w(O_t)
            if terminated or truncated or t is the final rollout transition:
                Ahat_t <- δ_t
            else:
                Ahat_t <- δ_t + γλAhat_t+1
            G_t <- Ahat_t + v_w(O_t)

    # Standardize advantages once and keep all targets fixed
    Ahat <- standardize(Ahat)

    # Reuse the same rollout for K optimization epochs
    for epoch <- 1,...,K:
        permutation <- random_permutation(rollout)
        for each minibatch of size M:
            log π <- corrected_log_prob(A_t, U_t, μ_θ(O_t), σ)
            ω_t <- exp(log π - old_log_π_t)

            # Compute clipped actor loss
            L_actor <- mean(-min(ω_t · Ahat_t, clip(ω_t, 1 - ε, 1 + ε) · Ahat_t))

            # Perform actor update (with clipping)
            Adam_actor.zero_grad()
            ∇L_actor <- backprop(L_actor)
            ∇L_actor <- ∇L_actor · min(1, 0.5 / ||∇L_actor||₂)
            Adam_actor.step()
            project(log standard deviations)

            # Compute critic loss and perform critic update (with clipping)
            L_critic <- mean(1/2 · (v_w(O_t) - G_t)²)
            Adam_critic.zero_grad()
            ∇L_critic <- backprop(L_critic)
            ∇L_critic <- ∇L_critic · min(1, 0.5 / ||∇L_critic||₂)
            Adam_critic.step()
        
        # Check policy drift after each complete epoch
        KL <- mean(approximate_KL(old_log_π, current_log_π))
        if KL > KL_target: break
```

Code references:
* [`PPO Agent`](../src/agents/ppo.py)
* [`PPO Training Engine`](../src/training/engines/ppo.py)

## Provenance of numerical choices

The following distinction is intentional:

| Choice | Origin |
|---|---|
| Policy-gradient, actor-critic, GAE and clipped-PPO equations | Course notes |
| $\gamma=0.9995$ | Racing timescale derivation in `MDP.md` |
| Actor widths | Scientific factor fixed by the experiment design |
| Fixed `(64, 64)` critic | Project control that prevents a critic-capacity confound |
| Adam $\beta_1$, $\beta_2$ and $10^{-8}$ | Defaults recommended in the original Adam paper |
| PPO rollout 2048, minibatch 64 and clip 0.2 | Starting configuration reported for continuous control in the original PPO paper |
| PPO 4 epochs and target KL 0.02 | Stability choice after observing repeated policy collapse under 10 unconditional epochs |
| $\lambda=0.95$ | Conventional GAE/PPO starting value |
| Squashed Gaussian, state-independent dispersion and its bounds | Explicit policy-class choice required by the bounded action space |
| Naive running-sum normalization | Explicit project simplicity choice |
| Initialization gains, observation clipping and gradient norm 0.5 | Explicit stability choices |
| REINFORCE batch of 8 and A2C rollout of 2048 | Explicit collection trade-offs, checked before reported runs |

Primary external references for choices absent from the course notes are:
* [GAE paper](https://arxiv.org/abs/1506.02438)
* [PPO paper](https://arxiv.org/abs/1707.06347)
* [Adam paper](https://arxiv.org/abs/1412.6980)
* [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html)
