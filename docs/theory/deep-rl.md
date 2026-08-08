# 7. Deep Reinforcement Learning

We now consider the most recent advances in the field of Reinforcement Learning.
We start with modern **Actor-Critic** methods, continuing on the previous material.

## 7.1 Advantace Actor-Critic (A2C)

**A2C** is the more famous version of the **V-Critic Actor-Critic** algorithm which we have already seen.
It integrates the standard **Actor-Critic** logic with the idea of **Synchronous Batched Updates**, similarly to REINFORCE.
The purpose is to reduce *noise* in the gradient estimates and with that the overall variance.

The algorithm works as follows:

1. Initialize the parameters $\mathbf{\theta}$ for the actor and $\mathbf{w}$ for the critic.
2. For each iteration:
    * Collect a rollout of $T$ timesteps, interacting with the environment using the current policy $\pi_{\mathbf{\theta}}$.
    A rollout typically represents a long trajectory or a sequence of variable-length episodes. 
    Assuming there to be multiple episodes, we denote $\tau_t$ the final timestep of the episode that is running at timestep $t$.
    * Compute **Returns-to-go**, which represent the **Critic Targets**, for each timestep $t$:
        $$ G_t = \sum_{k=t}^{\tau_t} \gamma^{k-t} R_{k+1} $$
    * Compute the **TD errors** based on the current critic $v_{\mathbf{w}}$:
    $$ 
    \delta_t^{\mathbf{w}} = 
    \begin{cases}
    R_{t+1} + \gamma v_{\mathbf{w}}(S_{t+1}) - v_{\mathbf{w}}(S_t) & \quad \text{if} \,\, t \neq \tau_t \\ 
    R_{t+1}- v_{\mathbf{w}}(S_t) & \quad \text{otherwise}
    \end{cases}
    $$
    notice that the *end of episodes* should be handled differently.
    * Define the **Actor Loss**:
        $$ \mathcal{L}_\text{actor}(\mathbf{\theta}) = - \sum_{t=0}^{T-1} \log{\pi_{\mathbf{\theta}}(A_t \mid S_t)} \delta_t^{\mathbf{w}} $$
    * Define the **Critic Loss**:
        $$ \mathcal{L}_\text{critic}(\mathbf{w}) = \frac{1}{2} \sum_{t=0}^{T-1} (v_{\mathbf{w}}(S_t) - G_t)^2
    * Run a step of **Gradient Descent** using **Adam Optimizer**, minimizing the loss:
        $$ \mathcal{L}_\text{actor}(\mathbf{\theta}) + \mathcal{L}_\text{critic}(\mathbf{w}) $$
        with respect to the parameters $\mathbf{\theta}$ and $\mathbf{w}$.
    > Actually, we do not really minimize the sum of the two losses. Instead, we minimize $\mathcal{L}_\text{actor}(\mathbf{\theta})$ fixing $\mathbf{w}$ and then minimize $\mathcal{L}_\text{critic}(\mathbf{w})$ fixing $\mathbf{\theta}$.

> Note that **A2C** is **On-Policy**, as we generate the data that we use to improve the policy using the same policy that we are optimizing. This is different from **DQN**, whose replay buffer makes it so that it's actually **Off-Policy** due to being used multiple times.

### A2C with Generalized Advantage Estimation (GAE)

Standard **A2C** uses the **One-Step TD Error** as its **Advantage Estimator**:
$$ \hat{\mathbb{A}}^{\text{A2C}}_t = \delta_t^{\mathbf{w}} = R_{t+1} + \gamma v_{\mathbf{w}}(S_{t+1}) - v_{\mathbf{w}}(S_t) $$

A better **Advantage Estimator** is that provided by the **GAE** variant of **A2C**, which basically combines multiple **TD Errors from multiple steps** into a single estimator for the advantage:
* Recall $k$-step returns:
    * $v_t^{(1)} = R_{t+1} + \gamma v_{\mathbf{w}}(S_{t+1}) $
    * $v_t^{(2)} = R_{t+1} + \gamma R_{t+2} + \gamma^2 v_{\mathbf{w}}(S_{t+2}) $
    * $\dots$
    * $v_t^{(k)} = R_{t+1} + \dots + \gamma^{k-1} R_{t+k} + \gamma^k v_{\mathbf{w}}(S_{t+k}) $
* Then, we define the $\lambda$-return:
    $$ v_t^\lambda = \sum_{k=1}^{\infty} \lambda^{k-1} v_t^{(k)} $$
    with $\lambda \in (0,1)$ doing the same job as in TD($\lambda$).
* Finally, we replace the **1-step return** with a **$\lambda$-step return** in the **Advantage Estimator**:
    $$ \hat{\mathbb{A}}^{\text{GAE}}_t = v_t^\lambda - v_{\mathbf{w}}(S_t) = \sum_{k=0}^{\infty} (\gamma \lambda)^k \delta_{t+k}^{\mathbf{w}} $$

The **A2C + GAE** Algorithm is now totally equivalent to the standard **A2C** algorithm, but replacing $\delta_t^{\mathbf{w}}$ with the new **GAE Advantage Estimator** $\hat{\mathbb{A}}^{\text{GAE}}_t$.
Of course, if we choose $\lambda = 0$ or $k = 1$ we fall back to ordinary **A2C**. This has higher *bias* and lower *variance*.
If instead we increase $\lambda$ towards $1$, we reduce the *bias* and increase *variance*.


## 7.2 (Deep) Deterministic Policy Gradient (DDPG and TD3)

We have already cited the idea of using **Policy Gradient** with a **Deterministic Policy**, explaining how we need to use a different estimator for the gradient of the Performance Function $J$.
Moreover, we have seen that these algorithms need a way to add **exploratory** behaviour to their interactions with the environment. 
This makes them clearly **Off-Policy**.

**DDPG** and **TD3** are the most famous implementations of such algorithms, with **TD3** being an improved, more stable version of **DDPG**.

## 7.3 Soft Actor Critic (SAC)

Starting from the **A2C Algorithm**, **SAC** chooses to make some modifications in order to favour exploration:
* **Exploration** can be measured by **Uncertainty** in the Policy values: the less certain they are about the action to pick, the more likely we are to explore new states.
* **Exploration is convenient**, as it allows to keep open paths that less-exploratory policies would immediately close up. Some of these paths could later reveal to be the best ones. This is why we care about **uncertainty*+ in our policy.
* To measure **uncertainty**, we use **Entropy**:
    $$ H(\pi(\cdot \mid s)) = - \sum_a \pi(a \mid s) \log{\pi(a \mid s)} $$
    The higher the entropy, the more stochastic the policy is, the more exploratory the policy is.

The idea of **SAC** is to simply have an **Entropy Bonus** factor that favours an exploratory behavior for the policy.
In practice, we adopt a **Soft $Q$-Function** that includes *entropy over future actions* when choosing how good a certain pair $(s, a
)$ is:
$$
q_\pi^{\text{soft}}(s, a) := r(s, a) + \mathbb{E} \left[ \sum_{t=1}^{\infty} \gamma^t \bigg( r(S_t, A_t) + \lambda H(\pi(\cdot \mid S_t)) \bigg) \bigg| \, S_0 = s, A_0 = a, \pi \right] $$

The parameter $\lambda$ becomes the balancer between too much entropy and standard A2C.

From the new **soft $Q$-function** derives a new expression of the policy and new expressions for updating parameters. 


## 7.4 Trust Region Methods: Sample Reuse

Up until now we have run **Actor-Critic** methods with the idea of collecting new samples, updating actor and critic using these samples, and then discarding them.
We now want to explore the possibility of **reusing samples** multiple times, instead of immediately discarding them.

The idea of reusing samples in an **Actor-Critic** dynamic has to deal with the issue that samples are related to the **policy** we are using to collect them. As we move on, updating the weights for our policy, the samples we have previously collected do not match the new **policy**.

This is why we introduce the idea of a **Trust Region**: if we limit policy updates to be relatively small, we can still use old generated data. Methods that implement this logic are called **Trust Region Methods**.


### Importance Sampling

A core concept for **Trust Region Methods** is the notion of **Importance Sampling**.
In general, Importance Sampling represents the idea of sampling from a distribution in an uneven way, giving a higher chance of being extracted to certain samples that are deemed *more important*.

Formally:
* Consider the idea of having some sample data $X$ from a distribution $q$, but we want to use it to estimate a certain expectation of $f(x)$ over a different distribution $p$:
    $$ X \sim q, \quad \mu = \mathbb{E}_{x \sim p} [f(x)] \quad \text{with} \quad p \neq q $$
    This is equivalent to our case, where we have collected samples with an old policy and now want to use them to compute an expectation over the current policy.
* What we can do is compute an **Importance-Weighted Estimate**:
    $$ \hat{\mu} = \frac{p(X)}{q(X)} f(X) $$
    This quantity weights each sample $x$ according to a weight of $\frac{p(X)}{q(X)}$.
* It can be shown that the estimation is unbiased:
    $$ \mathbb{E}[\hat{\mu}] = \sum_x q(x) \frac{p(X)}{q(X)} f(x) = \mathbb{E}_{x \sim p}[f(x)] = \mu $$

Unfortunately, **importance sampling** is theoretically unbiased but also **very noisy**. That is, if $p$ and $q$ are very different, the values of the weights will explode and the estimation will have a very high **variance**.

The solution to decrease **variance** in importance sampling is to use a **Surrogate Objective Function** $\mathcal{L}(\theta)$ that trades *variance* for some *bias*.

### Surrogate Loss for RL

In order to use **Importance Sampling** in the context of Reinforcement Learning we need to take some careful steps:

1. First, we focus on the quantity that we are trying to estimate (the $f$ function from before). Of course this is:
    $$ J(\theta') \quad \text{for the updated policy} \, \pi' $$
    The issue with this objective is that it is only related to $\pi'$, the latest policy. This means that all the samples collected with the previous policy $\pi$ cannot be used.  
    The solution is given by these two observations:
    * Maximizing the **Performance Function** $J(\theta')$ is equivalent to maximizing the **improvement** of $\pi'$ over $\pi$.
    * By the **Performance Difference Lemma**:
        $$ J(\theta') - J(\theta) = \mathbb{E}_{\substack{S \sim d_{\pi'} \\ A \sim \pi'(\cdot \mid S)}} [\mathbb{A}^\pi(S, A)] $$
    In other words, the **Average Advantage** of $\pi'$ over $\pi$ can be used to determine how much better $\pi'$ is than $\pi$.

2. Now, since our samples come from $\pi$ and not $\pi'$, we need to apply the **Performance Sampling**. This becomes:
    $$ J(\theta') - J(\theta) = \mathbb{E}_{\substack{S \sim d_{\pi} \\ A \sim \pi(\cdot \mid S)}} \left[
        \frac{d_{\pi'}(S)\pi'(A \mid S)}{d_\pi(S)\pi(A \mid S)} \mathbb{A}^\pi(S, A)
    \right] $$
    This is just the application of the performance sampling theory, which requires both the **state-observation functions** and the **policies themselves**.  
    We are presented with another problem: we do not know $d_\pi'$ nor $d_\pi$.
3. We make the decision (introducing bias) of assuming:
    $$ d_\pi' \approx d_\pi $$
    Now, numerator and denumerator simplify in the above estimate:
    $$ 
    J(\theta') - J(\theta) = \mathbb{E}_{\substack{S\sim d_{\pi}\\ A\sim\pi(\cdot\mid S)}}
    \left[
    \frac{\pi'(A\mid S)}{\pi(A\mid S)}
    \mathbb{A}^{\pi}(S,A)
    \right]
    $$
    However, this assumption does not come for free: for it to hold, $\pi'$ and $\pi$ should be **sufficiently similar**.
    To guarantee this, we add a penalty term to the objective $J(\theta') - J(\theta)$, which is higher the *more* different the two policies are. To do so, we use the **Total Variation Distance** between the two policies.
    The final result is the **Surrogate Loss**:
    $$
    \mathcal{L}(\pi')
    :=
    \mathbb{E}_{\substack{S\sim d_{\pi}\\ A\sim\pi(\cdot\mid S)}}
    \left[
    \frac{\pi'(A\mid S)}{\pi(A\mid S)}
    \mathbb{A}^{\pi}(S,A)
    \right]
    -
    \frac{2\gamma\epsilon}{(1-\gamma)^2}
    \mathbb{E}_{S\sim d_{\pi}}
    \left[
    D_{\mathrm{TV}}
    \left(
    \pi'(\cdot\mid S),
    \pi(\cdot\mid S)
    \right)
    \right]
    $$

In short, if we guarantee that $\pi'$ is somewhat close to $\pi$, then we can run **Gradient Descent** over the **Surrogate Loss** $\mathcal{L}(\pi')$ and obtain a result comparable to what we would get if we were to use just $J(\theta')$, but with the fundamental advantage of being able to use data collected from the interaction *multiple* times.

> Note that this is where the notion **Trust Region** comes from: for the whole dissertation above to hold, we need to modify $\pi$ such that $\pi'$ is within a region that we trust to be *not-too-different* from $\pi$. 

## 7.5 Trust Region Policy Optimization (TRPO)

The **Trust Region** logic requires to optimize the newly defined $\mathcal{L}(\pi')$.
Unfortunately, the **Total Variation Divergence** $D_{\mathrm{TV}}$ term is hard to optimize, as it is **non-differentiable**.

To get around this issue, the **TRPO Algorithm** does two things:

1. We replace the **Total Variation** term with an upperbound for it, given by the so-called **Pinker's inequality**:
    $$ D_{\mathrm{TV}}(\pi'(\cdot \mid s), \pi(\cdot \mid s)) \le \sqrt{\frac{1}{2}D_{\mathrm{KL}}(\pi'(\cdot \mid s), \pi(\cdot \mid s))} $$

2. Subsequently, we avoid the messy **penalization term** (which is now expressed using the **KL-Divergence**) and instead fix a constraint. The constraint states that the two policies should not differ by more than a term $\delta$:
    $$ \mathbb{E}_{S \sim d_\pi} [ D_{\mathrm{KL}}(\pi'(\cdot \mid S), \pi(\cdot \mid S))] \le \delta $$

Which means that in the end the **TRPO** algorithm optimizes only for the first term of the surrogate loss $\mathcal{L}(\pi')$, but making sure the constraint (2) is satisfied.

Unfortunately, the computation of **TRPO** is fairly tumultuous, involving the **Natural Gradient**, the **Fisher Matrix** and more... 
These are all necessary in order to perfor the **constrained update of the actor's parameters $\theta$**.

## 7.6 Proximal Policy Optimization (PPO)

**PPO** arises as a more *practical* alternative to **TRPO**. 
With **PPO**:
* We still replace the **TV** term with the Pinker's inequality, which expresses it through the **KL-Divergence**.
* We still avoid to have this term in the **Surrogate Loss** we are optimizing (in some versions of PPO we instead keep it):
    * In TRPO, the solution was to add a constraint that then required some complex calculations.
    * In PPO, the solution is simpler: we apply **clipping** to terms in the **Surrogate Loss**, so that we *approximately* limit updates to the **policy's parameters**.

More precisely, notice that the original **surrogate loss** contains:
$$ \frac{\pi'(A \mid S)}{\pi(A \mid S)} \mathbb{A}^\text{...}_t = \omega_{\theta' / \theta} \, \mathbb{A}^\text{...}_t  $$
where we assume a certain **Advantage Estimator** is used (e.g., **GAE**'s $\mathbb{A}^\text{GAE}_t$) and we rename the fraction term $\omega_{\theta' / \theta}$.

Then, **PPO clips exactly the term $\omega_{\theta' / \theta}$**:
$$ \mathrm{clip}(\omega_{\theta' / \theta}, 1 - \epsilon, 1 + \epsilon) $$

Which means that the **Surrogate Loss** employed by **PPO** is now:
$$
\mathcal{L}_{\mathrm{actor}}(\theta)
=
-\sum_{t=0}^{T-1}
\min\left\{
\omega_{\theta'/\theta}(S_t, A_t)\hat{\mathbb{A}}_t,\,
\operatorname{clip}\left(
\omega_{\theta'/\theta}(S_t, A_t),
1-\epsilon,
1+\epsilon
\right)\hat{\mathbb{A}}_t
\right\}.
$$

> Notice that the role of the **min** operator is to avoid that **clipping** simply avoids any abrupt change: we avoid abrupt changes towards *advantageous* directions, while allowing those that penalize the current policy.

The final **PPO Algorithm** becomes:
1. Initialize the parameters $\mathbf{\theta}$ for the actor and $\mathbf{w}$ for the critic.
2. For each iteration:
    * Collect a rollout of $T$ timesteps, interacting with the environment using the current policy $\pi_{\mathbf{\theta}}$.
    * Compute the **Critic Targets** $y_t$ and the **Advantage Estimates** $\hat{\mathbb{A}_t}$, using the current critic $v_{\mathbf{w}}$.
    * Define the **Actor Loss**:
        $$
        \mathcal{L}_{\mathrm{actor}}(\theta)
        =
        -\sum_{t=0}^{T-1}
        \min\left\{
        \omega_{\theta'/\theta}(S_t, A_t)\hat{\mathbb{A}}_t,\,
        \operatorname{clip}\left(
        \omega_{\theta'/\theta}(S_t, A_t),
        1-\epsilon,
        1+\epsilon
        \right)\hat{\mathbb{A}}_t
        \right\}.
        $$
    * Define the **Critic Loss**:
        $$
        \mathcal{L}_{\mathrm{critic}}(w)
        =
        \frac{1}{2}
        \sum_{t=0}^{T-1}
        (v_\mathbf{w}(S_t) - y_t)^2
        $$
    * Run a step (or multiple, in the **multi-epoch variant**) of **Gradient Descent** using **Adam Optimizer**, minimizing the loss:
        $$ \mathcal{L}_\text{actor}(\mathbf{\theta}) + \mathcal{L}_\text{critic}(\mathbf{w}) $$
        with respect to the parameters $\mathbf{\theta}$ and $\mathbf{w}$.
    > Again, we do not really minimize the sum of the two losses. Instead, we minimize $\mathcal{L}_\text{actor}(\mathbf{\theta})$ fixing $\mathbf{w}$ and then minimize $\mathcal{L}_\text{critic}(\mathbf{w})$ fixing $\mathbf{\theta}$.

The **Clipping** operation in **PPO** is the force that avoids too abrupt (optimistic) changes to $\theta$:
* It mitigates the **bias**, as it makes the assumption $d^{\pi'} \approx d^\pi$ reasonable.
* It mitigates the **variance**, thanks to the clipping of the importance weights.

## Conclusions:

| Algorithm     | Class                      | On vs Off Policy | Description                                                                                                                                                  | Reference Paper                                                                                                             |
| ------------- | -------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| **DQN**       | Value-based                | **Off-policy**   | Learns an action-value function $Q(s,a)$ with a neural network; uses experience replay and a target network to stabilize Q-learning.                         | Mnih et al., 2015 — *Human-level control through deep reinforcement learning*                                               |
| **REINFORCE** | Policy gradient            | **On-policy**    | Directly optimizes the policy using a Monte-Carlo estimate of the return as the policy-gradient weight.                                                      | Williams, 1992 — *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning*                |
| **A2C**       | Actor-Critic               | **On-policy**    | Combines a policy (actor) with a value-function critic; uses the 1-step TD error as an advantage estimate.                                                   | Mnih et al., 2016 — *Asynchronous Methods for Deep Reinforcement Learning*                                                  |
| **A2C + GAE** | Actor-Critic               | **On-policy***   | A2C using Generalized Advantage Estimation, which combines multiple TD errors to obtain a more favorable bias–variance trade-off for the advantage estimate. | Schulman et al., 2016 — *High-Dimensional Continuous Control Using Generalized Advantage Estimation*                        |
| **DDPG**      | Deterministic Actor-Critic | **Off-policy**   | Uses a deterministic policy gradient with an actor and critic, together with experience replay and target networks.                                          | Lillicrap et al., 2016 — *Continuous control with deep reinforcement learning*                                              |
| **TD3**       | Deterministic Actor-Critic | **Off-policy**   | Improves DDPG using two critics, delayed actor updates, and target-policy smoothing to reduce overestimation and instability.                                | Fujimoto et al., 2018 — *Addressing Function Approximation Error in Actor-Critic Methods*                                   |
| **SAC**       | Stochastic Actor-Critic    | **Off-policy**   | Extends actor-critic learning with a maximum-entropy objective, encouraging exploration while maximizing expected reward.                                    | Haarnoja et al., 2018 — *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor* |
| **TRPO**      | Trust-Region Actor-Critic  | **On-policy***   | Optimizes a surrogate objective while constraining the KL divergence between the old and new policies, limiting the size of policy updates.                  | Schulman et al., 2015 — *Trust Region Policy Optimization*                                                                  |
| **PPO**       | Trust-Region Actor-Critic  | **On-policy***   | Simplifies TRPO by replacing its explicit trust-region constraint with a clipped surrogate objective that discourages excessively large policy updates.      | Schulman et al., 2017 — *Proximal Policy Optimization Algorithms*                                                           |

\* Single-epoch versions. If the collected rollout is reused for multiple optimization epochs, the first epoch is on-policy, while subsequent epochs are technically off-policy with respect to the data-collection policy.