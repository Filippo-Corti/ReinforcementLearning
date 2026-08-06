# 5. Policy Gradient (Finite Horizon - Actor-Only)

With Value-Function Approximation we were dealing with the idea of building a parametrized function, either $\hat{v}_{\mathbf{w}}(s)$ or $\hat{q}_{\mathbf{w}}(s, a)$, that could represent a good approximation for the actual value-function $V$ of action-value-function $Q$ of the MDP.
Having found this approximation (that is, having found proper values for $\mathbf{w}$), we were then able to extract the policy as:
$$ \pi(s) = \text{arg} \max_a \hat{q}_{\mathbf{w}}(s, a) $$

This is the scenario we were at:
> Type: **Deep Learning**
>
> State Space: Continuous, $\mathcal{S} = (x, y) \in [0, W] \times [0, H]$
>
> Action Space: Discrete $\mathcal{A} = \{ 0, 1, 2, 3 \} $

Now, we consider a different approach to the problem: building a parametrized policy directly:
$$ \pi_{\mathbf{\theta}}(s), \quad \mathbf{\theta} \in \Theta \subseteq \mathbb{R}^d $$
The core idea is to optimize directly the parameters $\mathbf{\theta}$ of the policy, using a series of methods called **Policy-Gradient Methods**.

The main reason for this choice is that we can now consider a different scenario:
> Type: **Policy-Gradient Learning**
>
> State Space: Continuous, $\mathcal{S} = (x, y) \in [0, W] \times [0, H]$
>
> Action Space: Continuous $\mathcal{A} \subseteq \mathbb{R}^{d_\mathcal{A}} $

In fact, if we were to have a continuous *Action Space* in the DQN setting we would have trouble handling the expression $ \text{arg} \max_a \hat{q}_{\mathbf{w}}(s, a) $, as it would correspond to an optimization problem over the continuous space of actions.


## Formalization of the Policy-Gradient Algorithms

Let's consider this newly introduced scenario:
* Continuous State Space $\mathcal{S}$ and Action Space $\mathcal{A}$;
* Transition Kernel ${p(\cdot \mid s, a) : s \in \mathcal{S}, a \in \mathcal{A}}$;
* Reward Function $r : \mathcal{S} \times \mathcal{A} \rightarrow [-1, 1]$;
* Starting-state Distributio $p_0$.
* (For the moment) Finite Horizon $T \in \mathbb{N}$
* (In general) Policy $\pi(\cdot \vert s) \in \Delta(\mathcal{A}) $, a mapping between a state $s$ and a distribution over the actions.

With the idea of parametrized policies $\pi_{\mathbf{\theta}}$, we only consider the **Parametric Policy Space** (or *policy class*):
$$ \Pi_{\Theta} = \left\{ \pi_{\mathbf{\theta}} \mid \mathbf{\theta} \in \Theta \right\} \subseteq \Delta_{\mathcal{A}}^{\mathcal{S}} $$
where $ \mathbf{\theta} \in \Theta \subseteq \mathbb{R}^d $ is the parameter vector of the policy. For example, $\mathbf{\theta}$ can represent the weights of a neural network.
> Note: the mapping between the parameters $\mathbf{\theta}$ and the policies $\pi_{\mathbf{\theta}}$ is in general neither surjective nor injective.


Earlier, our objective was to find the Q-function approximation $\hat{q}_{\mathbf{w}}$ that better approximated the real Q-function (unknown).
Now, our objective can be expressed by the **Performance Function** $J : \Pi \rightarrow \mathbb{R}$:
$$ J(\pi_{\mathbf{\theta}}) = \mathbb{E} \left[ \sum_{t=0}^{T-1} r(S_t, A_t) \right] $$
This represents the expected return for a finite-horizon trajectory in the MDP, using the policy $\pi_{\mathbf{\theta}}$.

Now, we can represent the problem of finding the *best-in-class policy* for a given MDP as:
$$ \pi_{\mathbf{\theta}}^\star \in \text{arg} \max_{\pi_{\mathbf{\theta}} \in \Pi} J(\pi_{\mathbf{\theta}}) $$
Which we can express as a maximization w.r.t. the parameters $\mathbf{\theta}$ directly:
$$ \mathbf{\theta}^\star \in \text{arg} \max_{\mathbf{\theta} \in \Theta} J(\mathbf{\theta}) $$

> Unlike **DQN**, **Policy Gradient** chooses to optimize parameters $\mathbf{\theta}$ directly, using *stochastic gradient descent*.

## The REINFORCE Algorithm

Let's assume to have a *Policy Class* $\Pi_\Theta$ such that:
1. Policies are **Fully Mixed**:
$$ \pi_{\mathbf{\theta}}(a \mid s) > 0 \quad \forall s,a,\mathbf{\theta}$$
2. Policies are **Differentiabile**, w.r.t. the parameters:
$$ \nabla_{\mathbf{\theta}} \pi_{\mathbf{\theta}}(a \mid s) \quad \text{exists}$$

Then, since $\log\pi_{\mathbf{\theta}}(a \mid s)$ exists thanks to fully-mixed property, this also exists:
$$ \nabla_{\mathbf{\theta}} \log\pi_{\mathbf{\theta}}(a \mid s) = \frac{\nabla_{\mathbf{\theta}} \pi_{\mathbf{\theta}}(a \mid s)}{\pi_{\mathbf{\theta}}(a \mid s)} $$

This quantity is called the **Score Function** (or *Eligibility Vector*) and measures how sensitive the policy is to parameter changes, in terms of how much the action chosen by the policy would change in response.

We now introduce the first Policy-Gradient Algorithm: **REINFORCE**. \
The idea behind REINFORCE is straightforward:
1. Collect a batch of $n$ independent trajectories $\tau^1, \dots. \tau^n$ by playing the current policy $\pi_{\mathbf{\theta}_k}$.
2. Use that batch to run an update step of **Gradient Descent** and update the policy parameters: 
$$ \mathbf{\theta}_{k+1} \leftarrow \mathbf{\theta}_k + \alpha \hat{\nabla} J(\mathbf{\theta}_k) $$
3. Repeat for $k=0,1, \dots$.

It is a classic **SGD** loop on the Performance Function $J$, in order to find the parameters $\mathbf{\theta}$ that describe the policy that yields the *maximum expected returns*.
The parameter $\alpha > 0$ is the step size, or *learning rate*.

### Computing $\hat{\nabla} J(\mathbf{\theta}_k)$

An ideal SGD loop would perform its updates using $\nabla J(\mathbf{\theta}_k)$, the actual gradient of $J$ w.r.t. $\mathbf{\theta}$.
Instead, REINFORCE uses $\hat{\nabla} J(\mathbf{\theta}_k)$.

The reason is very simple: computing $\nabla J(\mathbf{\theta}_k)$ requires **full knowledge of the MDP**.
Of course, we assume that this information is not part of what we know.

As an alternative, REINFORCE uses the **REINFORCE Estimator**:

* First, let us observe that we can express the Performance Function $J$ using **Trajectories**. In fact:
    * Denote a *trajectory* by $\tau = \left( S_0, A_0, \dots, S_{T-1}, A_{T-1}, S_T \right)$. That is the sequence of states and actions observed during a (finite horizon) episode.
    * We can express the probability of observing the trajectory $\tau$ by playing the policy $\pi$ as:
    $$ p_\pi(\tau) = p_0(S_0) \cdot \prod_{t=0}^{T-1} \pi(A_t \mid S_t) p(S_{t+1} \mid S_t, A_t)$$
    * The return of the trajectory *tau* is of course:
    $$ R(\tau) = \sum_{t=0}^{T-1} r(S_t, A_t) $$
    * Having defined returns and probabilities, the **Performance Function** $J$ for a certain policy $\pi_{\mathbf{\theta}}$ is simply:
    $$ J(\mathbf{\theta}) = \mathbb{E}_{\tau \sim p_{\mathbf{\theta}}} \left[ R(\tau ) \right] $$


* Then, let us find the *actual* gradient of the Performance Function $J$:
$$ 
\nabla_{\mathbf{\theta}} J(\mathbf{\theta}) = \mathbb{E}_{\tau \sim p_{\mathbf{\theta}}} \left[ \left( \sum_{t=0}^{T-1} \nabla_{\mathbf{\theta}} \log{\pi_{\mathbf{\theta}}(A_t \mid S_t)} \right) R(\tau) \right]
$$ 

$$ \textit{\dots Proof of the Finite-Horizon Policy-Gradient Theorem \dots} $$


* From which we can immediately define the **REINFORCE Estimator**, given a trajectory $\tau = (S_0, A_0, \dots, S_T)$:
$$ g(\mathbf{\theta}; \tau) = \left( \sum_{t=0}^{T-1}{\nabla \log{\pi_{\mathbf{\theta}}(A_t \mid S_t)}} \right) \cdot R(\tau) $$

The **REINFORCE Estimator** is:
1. **Monte Carlo**, as it is a way to estimate a quantity $x$ which is defined as:
$$ x = \mathbb{E}[ \dots ] $$

2. **Unbiased**, as from our derivation of the estimator it is easy to see that:
$$ \mathbb{E}_{\tau \sim p_{\mathbf{\theta}}} \left[ g(\mathbf{\theta}; \tau) \right] = \nabla J(\mathbf{\theta}) $$

3. **Computable entirely from observations of the MDP**, without the need for any prior knowledge.

4. **Easy to use in practice**, as we just have to naturally interact with the environment using $\pi_{\mathbf{\theta}}$, as we have seen.


> Observe that $\hat{\nabla} J(\mathbf{\theta}_k)$ is nothing other than the batched version of the REINFORCE Estimator:
> $$\hat{\nabla} J(\mathbf{\theta}_k) = \frac{1}{n} \sum_{i=1}^{n}{g(\mathbf{\theta}; \tau^i)}$$
> where the trajectories $\tau^1, \dots, \tau^n$ are i.i.d. and $n$ is called the *batch size*.


## Policy-Gradient Algorithms: Pros and Cons

We compare Policy-Gradient Algorithms to other families of RL Algorithms, in particular **Value-Function Approximation**:

| #     | Advantage                                  | Description                                                                                                                                                                  |
| ----- | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1** | **Continuous actions**                     | Naturally support continuous action spaces.                                                                                                                                  |
| **2** | **Convergence guarantees**                 | Well-understood **local convergence guarantees**, even when function approximation is involved.                                                                              |
| **3** | **Robustness to noise**                    | Compared to value-based methods, actions (or action distributions) change less abruptly when states or parameters are perturbed. This typically yields more stable learning. |
| **4** | **Stochastic policies**                    | Naturally support stochastic policies, which are necessary in **partially observable** and **strategic** environments, while also providing a minimal amount of exploration. |
| **5** | **Less reliance on the Markov assumption** | Suffer less from violations of the Markov assumption since it is not exploited directly, at least in actor-only algorithms.                                                  |
| **6** | **Easy incorporation of domain knowledge** | The policy space can be designed to include only behaviors relevant to the application, e.g., controllers with a small number of tunable parameters.                         |
| **7** | **Safety**                                 | The policy space can be designed to exclude unsafe behaviors.                                                                                                                |

| #     | Disadvantage               | Description                                                                                                                                                                                                                                       |
| ----- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1** | **High variance**          | Gradient estimators tend to have high variance, making convergence slow and requiring large amounts of simulation data. Variance-reduction techniques can help mitigate this.                                                                     |
| **2** | **Bias**                   | The policy space may not include good policies if it is not designed carefully.                                                                                                                                                                   |
| **3** | **Local optima**           | Only convergence to **local optima** is guaranteed in general. Nonconvex optimization heuristics, such as random restarts, can be employed. Global convergence can be ensured in some special cases, e.g., **Linear Quadratic Regulators (LQR)**. |
| **4** | **Deterministic policies** | Do not naturally support deterministic policies, which can be problematic for safety-critical applications. Deterministic variants do exist.                                                                                                      |

## Variance Reduction Techniques

A big problem that **Policy-Gradient Algorithms** have to deal with is **Variance**. This is expected, as we are using a *Monte-Carlo* Approach.

In order to contain this issue, several variance reduction techniques have been explored:
