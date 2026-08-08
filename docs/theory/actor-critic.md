# 6. Policy Gradient (Actor-Critic)

The idea of **Actor-Critic Policy Gradient** is to employ **Value-Function Approximation** on the side, in order to:
* Reduce overall **variance**, at the cost of adding some *bias*.
* Fix the waste of interaction steps that we have seen in the Monte-Carlo Discounted Horizon Policy-Gradient Algorithm.

The idea is to have:

1. The **Policy** $\pi_{\mathbf{\theta}}$ be the **Actor**. It's the one we are trying to optimize by running a form of **Gradient Descent**.

2. The **Q-Function Approximation** $q_{\mathbf{w}}$ be the **Critic**. It's the same parametric value function approximator that we've seen when talking about VFA.

Let's start with the **Infinite Horizon Gradient of the Performance Function** we have previously identified:
$$
\nabla J(\mathbf{\theta}) = \frac{1}{1-\gamma}{ \mathbb{E}_{\substack{s \sim d^{\pi_{\mathbf{\theta}}}(s) \\ a \sim \pi_{\mathbf{\theta}}(\cdot \mid s)}}} \left[ \nabla \log{\pi_{\mathbf{\theta}}(a \mid s)} \mathbb{A}^{\pi_{\mathbf{\theta}}}(s, a) \right]
$$

We have seen that, in order to find an **estimator** for that gradient which we can use in practice to update our policy, we should find a way to estimate the **Advantage**:
$$ \mathbb{A}^\pi(s,a) = Q^\pi(s,a) - V^\pi(s) $$

Up until now:
* We observed that $V^\pi(s)$ could just be interpreted as a baseline, and we generalized it to $b(s)$.
* We of course did not know the real value for $Q^\pi(s,a)$, so we adopted a Monte-Carlo approach and estimated it as:
    $$ Q^\pi(s,a) \approx G $$
    with $G$ being the return we observe during a random episode.
* In the end, we had a **gradient estimator** that looked like this:
    $$
        \hat{\nabla} J(\mathbf{\theta}) = \frac{1}{1-\gamma} \nabla \log{\pi_{\mathbf{\theta}}(A_T \mid S_T)} \left( G - b(S_T) \right)
    $$

We have seen, however, that this needed **long rollouts**, which were bad for two reasons:
* It takes a lot of time to collect, and the first $T$ iterations are completely wasted just to have a proper sample of $d^\pi$.
* It creates high variance.

## Q-Critic Actor-Critic 

The idea of **Actor-Critic** Methods is simple: instead of using samples $G$ as approximations for $Q^\pi(s, a)$ we use a Q-Function Approximation $q_\mathbf{w}$, which we progressively update to be a better approximation of the real $Q$-Function.

The new **Actor-Critic Estimator** is:
$$
\hat{\nabla} J(\mathbf{\theta}) = \frac{1}{1-\gamma} \nabla \log{\pi_{\mathbf{\theta}}(A_T \mid S_T)} \left( G - b(S_T) \right)
$$

**VFA** introduces *bias* into the estimator, but has the advantage of reducing the high *variance* that comes with it.

The dynamic is as follows:
* The **Actor** (the policy) tries to improve by gradient descent over its **parameters** $\mathbf{\theta}$. To do so, it relies on the current approximation $q_\mathbf{w}$ of the $Q$-function, which is used to build the gradient estimator $\hat{\nabla} J(\mathbf{\theta})$.
* The **Critic** (the q-function) improves its parameters $\mathbf{w}$ alongside the actor, *chasing* the value of the $Q$-function for the current policy (which is the *actor* and therefore keeps changing!).

A basic **Actor-Critic** algorithm, using the $Q$-Function as critic would look like this:
1. Start from a starting state $S_t = S_0 \sim p_0$.
2. Play $A_t \sim \pi_{\mathbf{\theta}_t}(\cdot \mid S_t)$ and observe the transition reward $r(S_t, A_t)$ and the next state $S_{t+1}$.
3. **Update the Actor** using the **Actor-Critic Estimator**:
    $$ \theta_{t+1} \leftarrow \theta_k + \alpha \frac{1}{1-\gamma} \nabla_{\mathbf{\theta}} \log_{\pi_{\mathbf{\theta}_t}}(A_t \mid S_t)(q_\mathbf{w}(S_t, A_t) - b(S_t)) $$
4. **Update the Critic**, using a TD Target for $Q^\pi(S_t, A_t)$:
    $$ y_t = r_t + \gamma q_{\mathbf{w}_t}(S_{t+1}, A_{t+1})$$
    $$ \mathbf{w}_{t+1} \leftarrow \mathbf{w}_t + \beta (y_t - q_{\mathbf{w}_t}(S_t, A_t)) \nabla_{\mathbf{w}} q_{\mathbf{w}_t}(S_t, A_t) $$
    which results from the gradient of the known loss function for TD learning:
    $$ L(\mathbf{w}) = \frac{1}{2}[y_t - q_{\mathbf{w}}(S_t, A_t)]^2 $$
5. Repeat from step 2.


## V-Critic Actor-Critic (Advantage Estimation)

An alternative formulation of the **Actor-Critic** dynamic uses the $V$-function as the **critic**, instead of the $Q$-function.
In other words, the **Q-Critic** used this Advantage Estimator:
$$ \mathbb{A} \approx q_\mathbf{w}(S_t, A_t) - b(S_t) \quad \text{with} \quad b(S_t) \approx V^\pi(S_t)$$
Instead, the **V-Critic** uses this one:
$$ \mathbb{A} \approx r_t + \gamma v_\mathbf{w}(S_{t+1}) - v_\mathbf{w}(S_t) $$
This is of course another *biased* estimator, but sufficiently good if $v_\mathbf{w}$ is a good approximation for the actual $V^{\pi_{\mathbf{\theta}}}$.

The main reason to make such choice is that we now do not need to sample $A_{t+1}$ and instead implicitly average over the next actions, via the $V$-function.

The **V-Critic Actor-Critic** algorithm looks like this:
1. Start from a starting state $S_t = S_0 \sim p_0$.
2. Play $A_t \sim \pi_{\mathbf{\theta}_t}(\cdot \mid S_t)$ and observe the transition reward $r(S_t, A_t)$ and the next state $S_{t+1}$.
3. Compute the **Advantage Estimator**:
    $$ \tilde{\delta_t} = r(S_t, A_t) + \gamma v_{\mathbf{w}_t}(S_{t+1}) - v_{\mathbf{w}_t}(S_t) $$
4. **Update the Actor**:
    $$ \theta_{t+1} \leftarrow \theta_k + \alpha \frac{1}{1-\gamma} \nabla_{\mathbf{\theta}} \log_{\pi_{\mathbf{\theta}_t}}(A_t \mid S_t)\tilde{\delta_t} $$
5. **Update the Critic**:
    $$ \mathbf{w}_{t+1} \leftarrow \mathbf{w}_t + \beta \tilde{\delta_t} \nabla_{\mathbf{w}} v_{\mathbf{w}_t}(S_t) $$
6. Repeat from step 2.

This is a simplification of the **A2C** Algorithm (**Advantage Actor Critic** Algorithm), covered later on.

> Note that it is a good practice to use $\alpha \ll \beta$, as we want our policy (the actor) to update more slowly than the critic. This is because the **critic** is fundamentally *chasing* the **actor**, and for it to keep up it should be a faster learner than the actor itself.

> Also note that both the Q-Critic and V-Critic implementations of the **Actor-Critic** logic use **biased policy-gradient estimators**. 
> There are two main reasons for this:
> * The estimator involves the critic, which is updated in a TD-learning fashion and therefore with high bias.
> * The states are distributed according to the state-occupancy measure (assuming we run indefinitely), but they are not in an independent order.

## Compatible Critic Actor-Critic

Finally, let's consider an **Actor-Critic** implementation which uses an **Unbiased Actor-Critic Policy-Gradient Estimator**.
This is a fairly weird asking, as **Actor-Critic** requires $Q^\pi$, which we typically only estimate in some *biased* way.
However, a **Compatible Critic** can be defined so that the approximation error (the bias) disappears.
For simplicity we do not consider the *baseline*, but we can still include it as it does not contribute to the bias.

In order to achieve an **unbiased estimator** we need to set two specific conditions:
1. **The features used by the critic must be exactly the score-function features used by the actor**:
    $$ \nabla_{\mathbf{w}} q_{\mathbf{w}}(s, a) = \nabla_{\mathbf{\theta}} \log \pi_{\mathbf{\theta}}(a \mid s) $$

    This condition can be satisfied by using a *linear VFA*, with the score function as features:
    $$ q_{\mathbf{w}}(s,a) = \mathbf{w}^\top \nabla_{\mathbf{\theta}} \log \pi_{\mathbf{\theta}}(a \mid s) $$

2. **The product between the critic error and the policy-gradient feature is expected to be zero**:
    $$ \mathbb{E}_{\substack{s \sim d^{\pi_{\mathbf{\theta}}} \\ a \sim \pi_{\mathbf{\theta}}(\cdot \mid s)}} \left[ (Q^\pi_{\mathbf{\theta}}(s, a) - q_{\mathbf{w}}(s, a)) \cdot \nabla_{\mathbf{w}} q_{\mathbf{w}}(s,a) \right] = 0 $$

    This condition can be satisfied by minimizing the *MSE* for the critic parameters, solving:
    $$ \mathbf{w} \in \text{arg} \min_{\tilde{\mathbf{w}}\in \mathbb{R}^{d_w}} \mathbb{E}_{\substack{s \sim d^{\pi_{\mathbf{\theta}}} \\ a \sim \pi_{\mathbf{\theta}}(\cdot \mid s)}} \left[ (Q^{\pi_{\mathbf{\theta}}}(s,a) - q_{\mathbf{w}}(s, a))^2 \right] $$

Under these two conditions, the standard **Actor-Critic Policy Gradient Estimator** becomes **unbiased**:
$$
\nabla J(\mathbf{\theta}) = \frac{1}{1-\gamma}{ \mathbb{E}_{\substack{s \sim d^{\pi_{\mathbf{\theta}}}(s) \\ a \sim \pi_{\mathbf{\theta}}(\cdot \mid s)}}} \left[ \nabla \log{\pi_{\mathbf{\theta}}(a \mid s)} q_{\mathbf{w}}(s, a) \right]
$$

# Deterministic Policy Gradient

We conclude the introduction to **Policy-Gradient** methods by observing that **Policy-Gradient** is also possible when the Policy is **Deterministic**.
In such cases:

* The $Q$-function **critic** should be differentiable with respect to parameters *and also* actions.
* Some form of **external exploration** is employed (*i.e.*, we do **off-policy** policy-gradient), as the policy is naturally non-exploratory.
* The **Gradient of the Performance Function** looks like this:
    $$ \nabla_{\mathbf{\theta}} J(\theta) = \frac{1}{1-\gamma} \mathbb{E}_{s \sim d^{\mu_{\mathbf{\theta}}}} \left[ \nabla_{\mathbf{\theta}} \mu_{\mathbf{\theta}}(s) \nabla_a Q^{\mu_{\mathbf{\theta}}}(s, a) \vert_{a = \mu_{\mathbf{\theta}}(s)} \right] $$