# 1. Gymnasium

> Type: **Tabular Learning**
>
> State Space: Discrete, $\mathcal{S} = \{ 0, 1, ..., H \cdot W-1 \}$
>
> Action Space: Discrete $\mathcal{A} = \{ 0, 1, 2, 3 \} $

## The Gymnasium Library

`Env` is the central environment class. It exposes:
* `obs, info = env.reset(seed=42)`, to start a new episode;
* `obs, reward, terminated, truncated, info = env.step(action)`, to advance the episode by 1 timestep;
* `frame = env.render()`, to render the env.

Both `reset` and `step` return an *observation* of the *state*. 
This coincides with the state only for **Fully-Observable MDPs**.
In practice, we most often have **Partially-Observable MDPs** (POMDPs).

The `step` method returns *two* flags:
* `terminated` signals that the new state is terminal.
* `truncated` signals that the episode was cut short by an external mechanism.

You register a new `Env` using `gymnasium.make("env-id")`.

| MDP element | Gymnasium representation |
|---|---|
| State space $\mathcal{S}$ | `env.observation_space`, an instance of a `Space` (e.g. `Discrete`, `Box`) |
| Action space $\mathcal{A}$ | `env.action_space`, an instance of a `Space` |
| Initial state distribution $\rho_0$ | encoded inside `env.reset(seed=...)`, which returns the initial observation |
| Transition kernel $P(s' \mid s, a)$ | encoded operationally inside `env.step(a)`; only sampled, not queryable |
| Reward function $R(s, a, s')$ | returned by `env.step(a)` as the immediate reward $r$ |
| Episode termination ($s$ terminal) | `terminated` flag returned by `env.step(a)` |
| Discount factor $\gamma$ | **not part of the environment** — owned by the agent |

# 2. Tabular Learning

## Tabular Q-Learning

We estimate the **discounted return** by taking action $a$ in state $s$, which is:

$$
q_\star(s, a) = \mathbb{E}\!\left[
    \, \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \,\Big|\, S_t = s, A_t = a, \pi_\star 
    \right]
$$

Since $\mathcal{S}$ and $\mathcal{A}$ are finite, we can represent this estimate $q_\star$ as a table:

$$
\mathcal{Q} \in \mathbb{R}^{|S| \times |A|}
$$

### Update Rule

We update the table iteratively using **temporal-difference learning**.
After observing a transition $(S_t, A_t, R_{t+1}, S_{t+1})$, we update:

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ \underbrace{R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a')}_{\text{TD target}} - Q(S_t, A_t) \right]
$$

### Properties

1. It is **sample-based**, as it does not require an explicit representation of the transition kernel or the reward function.
2. It is **off-policy**, as the **TD target** is build using a *greedy* policy, while the transition is produced using a *behavior policy* ($\epsilon$-greedy).
3. If $S_{t+1}$ is terminal, the **bootstrap term vanishes**. That is, the TD target is only $R_{t+1}$.

Under standard conditions:
* Finite MDP,
* Every state-action pair visited infinitely often,
* Learning rate $\alpha_t$ satisfying the Robbins-Monro conditions:
 $$\sum_t \alpha_t = \infty \quad \sum_t \alpha_t^2 < \infty$$
* GLIE policy

Q-learning's table converges almost surely to $q_\star$.

### Measuring RL Training

One training episode comprises:
* An initial `obs, _ = env.reset()`.
* A series of time steps, at each of which:
    * Agent picks an action `action`.
    * The environment executes the action with `next_obs, reward, ... = env.step(action)`.
    * The agent updates its Q-table using `obs, action, reward, next_obs`.
    * `obs = next_obs`.
* A termination, due to `terminated || truncated`.

During training, we want to know what to observe:
* The **return** for each episode. It measures worse-than-optimal returns, because the agent is using an $\epsilon$-greedy policy during the episodes. Still, it gives an idea of whether the agent is learning something or not.
* The **evaluation rollouts**, which measure how the agent performs acting **greedily** over a batch of episodes. This tracks learning in a proper way.
* The **exploration schedule**, which tracks how $\epsilon$ decreases over time. This is necessary to contextualize the **return** curve, as it tells how much exploration the agent is doing at the time.

## Tabular SARSA

We use the same idea of Q-Learning:

$$
q_\star(s, a) = \mathbb{E}\!\left[
    \, \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \,\Big|\, S_t = s, A_t = a, \pi_\star 
    \right]
$$

### Update Rule

However, now we use a different update rule
After observing a transition $(S_t, A_t, R_{t+1}, S_{t+1})$, we update:

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ \underbrace{R_{t+1} + \gamma \, Q(S_{t+1}, A_{t+1})}_{\text{TD target}} - Q(S_t, A_t) \right]
$$

We have replace the $\max_{a'} Q(S_{t+1}, a')$ term with $Q(S_{t+1}, A_{t+1})$.
SARSA uses for the **TD target** the exact policy that it uses to explore.
In fact, SARSA is **on-policy**.

### Q-Learning vs SARSA

To the limit, SARSA and Q-Leraning both converge to the target $q_\star$, however they do it by taking two different routes:
* SARSA is *more conservative*, as it evaluates states for how good they are now, not how good they could be if we used a greedy policy. It "knows" the risk of its exploratory policy.
* Q-Learning is *more optimistic*, as it ignores the fact that it's exploring. This makes convergence faster.

> In short, Q-Learning evaluates a greedy policy, while behaving according to an $\epsilon$-greedy one.
> Instead, SARSA evaluates the $\epsilon$-greedy policy directly.

#### Different Variances during Training

When we refer to **variance**, we should distinguish clearly which notion of variance we are referring to:

1. **Variance across evaluation rollouts, using the same policy**. This is measured within a rollout checkpoint. It is lower in SARSA as it is more cautious and takes safer routes (thus less risk -> less chance of a very different reward).
2. **Variance across checkpoints, during training**. This is measured between consecutive checkpoints. It is higher in SARSA, because it converges more slowly than Q-Learning.
3. **Variance across independent training runs**. This should always be measured, instead of relying on a single run.
4. **Variance across learning targets**. This is the between different (TD) targets during training. It is a property of the *learning algorithm*:
    * **Temporal-Difference Learning** (including both Q-Learning and SARSA) has a low *variance* across targetes, and a high *bias*. This is due to using only the one-step reward to build the TD-target: we give more weight to our current belief (the bootstrap term - the bias) and less weight to what we have just observed.

    $$T_t^{\text{TD}} = R_{t+1} + \gamma \cdot \texttt{[next-state value]}.$$

    * Monte-Carlo Learning** has a high *variance* across targets, and a low *bias*. This is due to using full sequences of discounted rewards as targets for the learning. These do not rely on current belief (low bias) but can vary significantly (high variance):
    
    $$T_t^{\text{MC}} = G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots + \gamma^{T-t-1} R_T,$$

## Monte Carlo Methods

As introduced above, Monte Carlo Methods estimate $q^{\pi}\left(s, a \right)$ by averaging actual *returns* observed under the policy $\pi$ (and not averaging TD targets!).
The estimator used by Monte Carlo Methods is therefore a simple **sample mean**.

To update the Q-table, we can use a running average by simply maintaining the number of times we have updated each pair $N(s, a)$:

$$N(s, a) \leftarrow N(s, a) + 1, \qquad Q(s, a) \leftarrow Q(s, a) + \frac{1}{N(s, a)} \big[ G - Q(s, a) \big],$$

Given a trajectory, sampled by using our policy for a full episode until termination:
$$ (S_0, A_0, R_1, S_1, A_1, R_2, \dots, S_{T-1}, A_{T-1}, R_T) $$
we can build a series of targets $G_t$ simply by walking the trajectory backwards:
$$ G_t = R_{t+1} + \gamma G_{t+1}, \quad t = T-1, T-2, ..., 0 $$
with $G_T = 0$.

Note that we can choose:
* **Every-visit MC**, which uses a $G_t$ for every time the pair $(S_t, A_t)$ is visited in the trajectory.
* **First-visit MC**, which only uses the $G_t$ fo the first time the pair $(S_t, A_t)$ has been visited.
From a theoretical standpoint, *every-visit MC* introduces bias due to different targets generated from the same trajectory for the same pair are correlated. In practice, they both converge more often than not.

### Q-Learning vs SARSA vs Monte Carlo

**On-policy or off-policy?** SARSA and MC evaluate the policy they actually follow during training; Q-learning evaluates the greedy policy regardless of what the agent is doing. 
Off-policy methods learn the best policy *as if* the agent were already optimal, which is what we usually want at convergence - but during training, they ignore the cost of the exploration that is happening. 
On-policy methods learn the value of the current behaviour including its exploration, which makes them more *robust during training* on stochastic problems but slightly more *conservative at convergence*.

**Bootstrap or full return?** TD methods (both Q-learning and SARSA) estimate from one-step transitions, building each target on top of the current Q-estimate. MC waits for an entire episode and uses the realised return. 
TD has lower variance and propagates information immediately between adjacent state-action pairs (the bootstrap term is what breaks the cycles we saw in MC); MC is unbiased and immune to bootstrapping artifacts. 
Bootstrapping is the more powerful tool whenever propagation between neighbouring states matters, which on most problems is essentially always.
