# 5. Policy Gradient (Infinite Horizon - Actor-Only)

Let us now consider **Policy Gradient Algorithms**:
1. **Infinite Horizon**: unlike earlier, we do not have a fixed number of timesteps $T$.
2. **Actor-Only**: same as earlier, we only do policy approximation (no VFA - we'll get there later).

The new formalization is a **Discounted** MDP:
* Continuous State Space $\mathcal{S}$ and Action Space $\mathcal{A}$;
* Transition Kernel ${p(\cdot \mid s, a) : s \in \mathcal{S}, a \in \mathcal{A}}$;
* Reward Function $r : \mathcal{S} \times \mathcal{A} \rightarrow [-1, 1]$;
* Starting-state Distribution $p_0$.
* **Infinite Horizon**.
* A class of parametric, stochastic, Markov Policies:
$$ \Pi_\Theta = \left\{ \pi_{\mathbf{\theta}} \in \Delta_{\mathcal{A}}^{\mathcal{S}} \mid \mathbf{\theta} \in \Theta \subseteq \mathbb{R}^d \right\} $$


The **Performance Function**, given by the *expected discounted return of the policy**, is now written as:
$$ J(\mathbf{\theta}) = \mathbb{E} \left[ \sum_{t=0}^{\infty}{\gamma^t r(S_t, A_t)} \right] = \frac{1}{1-\gamma} \mathbb{E} \left[ r(S, A) \right] $$
$$ \textit{\dots Proof of the above equivalence is in the notes \dots} $$

Now, in order to compute $J$, we would ideally do something like:
$$ J(\mathbf{\theta}) = \lim_{T\rightarrow \infty}{\frac{1}{T} \mathbb{E} \left[ \sum_{t=0}^{T-1}{r_t} \right]} $$
which corresponds to the average reward in an infinite horizon. 
Notice that, unlike the finite horizon, we cannot simply observe a full episode and then estimate $J$, as there is notion of *"full"* episode.

The solution comes from introducing a new element, the **State-Occupancy Measure** $d^{\pi_{\mathbf{\theta}}}$ induced by the policy $\pi_{\mathbf{\theta}}$:
$$ d^{\pi_{\mathbf{\theta}}}(s) = (1-\gamma) \sum_{t=0}^{\infty} \gamma^t \mathbb{P}(S_t = s) $$
This represents the probability of being in a certain state $s$ when interacting indefinitely with the MDP. The parameter $\gamma$ is the discount factor, corresponding to the probability of continuing the interaction (thus $1-\gamma$ is the probability to stop after each action).

Our new plan is:
1. Find a way to sample from $d^{\pi_{\mathbf{\theta}}}$.
2. Use that sampling to find an unbiased estimator for $\nabla J(\mathbf{\theta})$, just as for the finite horizon.

## Sampling from the State-Occupance Measure

We have seen two ways to sample from $d^{\pi_{\mathbf{\theta}}}$:

1. **Random Stopping**: Start from a starting state $S_t = S_0 \sim p_0$, then: 
    * With probability $1-\gamma$, stop and return $S_t$.
    * Otherwise, pick an action $A_t \sim \pi_{\mathbf{\theta}}(\cdot \mid S_t)$ and execute it getting $S_{t+1} \sim p(\cdot \mid S_t, A_t)$.
    * Repeat indefinitely.

2. **Random Horizon**: Start from a starting state $S_t = S_0 \sim p_0$ and a set horizon $T \sim \text{Geom}(1-\gamma)$, then:
    * Pick an action $A_t \sim \pi_{\mathbf{\theta}}(\cdot \mid S_t)$ and execute it getting $S_{t+1} \sim p(\cdot \mid S_t, A_t)$.
    * Stop at timestep $T$, return $S_T$.

> Note that $X \sim \text{Geom}(p)$ is a distribution so that $\mathbb{P}(X=n) = (1-p)^n p$.

These two approaches give theoretical guarantees that we have a way to sample independent samples from the State-Occupancy Measure $d^{\pi_{\mathbf{\theta}}}$. This, however, requires generating a full trajectory for each sample $s$.
In practice, we tend to avoid this and simply interact with the MDP (just like in the *finite horizon* case), ignoring the dependency between samples.


## Discounted Horizon Policy-Gradient Estimation

First, we introduce the **Advantage Function** $\mathbb{A}^\pi$:
$$ \mathbb{A}^\pi(s, a) = Q^\pi(s, a) - V^\pi(s) $$
The Advantage Function represents the difference between the $Q$-value function and the $V$-function (which, importantly, represents the average value given by playing the *stochastic* policy $\pi$ in $s$ and is not the $\max_a Q$).
More specifically, we use it to measure the advantage of playing action $a$ in a certain state $s$ instead of following the policy $pi$. 

The **Advantage Function** has two important properties:
1. $\mathbb{E}_{a \sim \pi(\cdot \mid s)} \left[ \mathbb{A}^\pi(s,a) \right] = 0 \quad \forall s, \pi$.
2. If $\pi^\star$ is an optimal policy, then $\mathbb{A}^{\pi^\star}(s,a) \le 0 \quad \forall s, \pi$. As a small note, notice that this may not hold for the best-in-class policy.

Now, we can use $\mathbb{A}^\pi$ to express an **Estimator** of the **Gradient** of the **Performance Function** in the **Discounted Horizon** setting:
$$
\nabla J(\mathbf{\theta}) = \frac{1}{1-\gamma}{ \mathbb{E}_{\substack{s \sim d^{\pi_{\mathbf{\theta}}}(s) \\ a \sim \pi_{\mathbf{\theta}}(\cdot \mid s)}}} \left[ \nabla \log{\pi_{\mathbf{\theta}}(a \mid s)} \mathbb{A}^{\pi_{\mathbf{\theta}}}(s, a) \right]
$$
or equivalently:
$$
\nabla J(\mathbf{\theta}) = \frac{1}{1-\gamma}{ \mathbb{E}_{\substack{s \sim d^{\pi_{\mathbf{\theta}}}(s) \\ a \sim \pi_{\mathbf{\theta}}(\cdot \mid s)}}} \left[ \nabla \log{\pi_{\mathbf{\theta}}(a \mid s)} Q^{\pi_{\mathbf{\theta}}}(s, a) \right]
$$
as the $V$-function in $\mathbb{A}$ does not contribute to the expectation.
Actually, the $V$-function can just be seen as a type of baseline for **variance reduction**. In general we could say:
$$
\nabla J(\mathbf{\theta}) = \frac{1}{1-\gamma}{ \mathbb{E}_{\substack{s \sim d^{\pi_{\mathbf{\theta}}}(s) \\ a \sim \pi_{\mathbf{\theta}}(\cdot \mid s)}}} \left[ \nabla \log{\pi_{\mathbf{\theta}}(a \mid s)} \left( Q^{\pi_{\mathbf{\theta}}}(s, a) - b(s) \right) \right]
$$

## Discounted Setting Monte-Carlo Policy Gradient Algorithm

Putting everything together we have defined an algorithm for the infinite horizon, similar to REINFORCE but with a different estimator:

1. Compute the **Discounted Setting Monte-Carlo PG Estimator** $\hat{\nabla} J(\mathbf{\theta})$ by doing the following:
    * Sample a $T \sim \text{Geom}(1-\gamma)$ and $S_0 \sim p_0$.
    * Play $T$ actions, each time picking $A_t \sim \pi_{\mathbf{\theta}}(\cdot \mid S_t)$ and observing $S_{t+1}$.
    * Sample a new horizon $\Delta \sim \text{Geom}(1-\gamma)$ and set $G \leftarrow 0$.
    * Play $\Delta$ actions, this time keeping track of the randomly truncated return by doing $G \leftarrow G + r(S_t, A_t)$ at each step.
    * Compute:
        $$
        \hat{\nabla} J(\mathbf{\theta}) = \frac{1}{1-\gamma} \nabla \log{\pi_{\mathbf{\theta}}(A_T \mid S_T)} \left( G - b(S_T) \right)
        $$
        eventually using a baseline $.

2. Use the estimate to update the policy parameters: 
$$ \mathbf{\theta}_{k+1} \leftarrow \mathbf{\theta}_k + \alpha \hat{\nabla} J(\mathbf{\theta}_k) $$

3. Repeat for $k=0,1, \dots$.

Our result is mathematically sound but very *impractical*:
* It has **very high variance**, due to the high variance of the $Q$-function estimates.
* Many interactions with the environment are needed but simply discarded. We only account for whatever happens after timestep $T$.
* It works in an episodic way, which may not fit every application.

A much more *practical* alternative is to run the **infinite horizon** just like the **finite horizon**, but adding a truncation to the trajectories.
Then, the return $G$ is computed as a **discounted sum**, rather than the actual sum as we have done above.
This is a sort of **Infinite Horizon REINFORCE**, providing an *approximation* and not a solid solution:

$$ G_t = \sum_{k=t}^{T_{\max}}{\gamma^{k-t}r_k} $$
$$ \mathbf{\theta}_{k+1} \leftarrow \mathbf{\theta}_k + \alpha \sum_{t}{\nabla_{\mathbf{\theta}}\log \pi_{\mathbf{\theta}}(A_t \mid S_t)} (G_t - b(S_T)) $$

