# 3. Value Function Approximation

> Type: **VFA**
>
> State Space: Continuous, $\mathcal{S} = (x, y) \in [0, W] \times [0, H]$
>
> Action Space: Discrete $\mathcal{A} = \{ 0, 1, 2, 3 \} $

# Preamble: dealing with continuous-state environments

If we consider the state to be a continuous location in a bounded rectangle, two consequences happen:

1. **States can be *similar* to others**. In the discrete environment, each state pair was equally dissimilar. Now that states are in a continuum, two close points are two *similar* states.

2. **States are *infinitely* many**. We cannot represent all states in an indexed table, because there is no possible enumeration.

One way to deal with consequence (2) is to discretize the state space into discrete cells. This way, we are back to a Tabular Learning scenario. This, however, encounters two issues:
* **Observations are not Markov**. Two episodes that bring us to the same (bin) state via different trajectories will likely bring us to different *continuous* locations, therefore they can lead to different future observations.
* **Q-Learning convergence is not guaranteed anymore**. This is because we are now modeling a *POMDP*, for which standard convergence conditions do not necessarily hold.

To solve these issues and the whole problem in general we should make explicit use of the fact that states can be *similar* (consequence (1)).

## VFA

**Value Function Approximation** is the family of methods that represent $\mathcal{V}$ or $\mathcal{Q}$ as a parametric function rather than a table.

The simplest case is the Linear Value Function Approximation:
$$ \hat{v}(s, \mathbf{w}) = \mathbf{w}^\top \mathbf{x}(s) $$
where $\mathbf{x}(s)$ is the *feature extractor* function, determining which features of the observation are a relevant part of the approximated $\hat{v}$ function. 

Choosing meaningful features is very much non-trivial.
In fact, it is possible that our choice of features automatically excludes the optimal function from the considered space. 
For example, the trivial choice $\mathbf{x}(s)=s$ would be technically possible but would only allow $\hat{v}$ to have three degrees of freedom, which can only represent functions that increase or decrease linearly in each direction. It's a matter of **representation capacity**.

In general, the idea is to replace the Q-table with a *parametric function*:
$$ 
\mathcal{Q} : \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}
$$ 
We therefore choose a family of functions $\hat{q}(s,a;\textbf{w})$, parametric by $\textbf{w} \in \mathbb{R}^{d}$, and we learn the value of $\textbf{w}$ so that $\hat{q}_{\textbf{w}}$ better approximates the real, unknown, $\mathcal{Q}$ function.

> The biggest advantage is *generalization*. By updating $\hat{q}_{\textbf{w}}$ in a certain point $(s, a)$ (a certain state), we also shape the function for the points near $(s, a)$. 
> We are making explicit use of the *similarity* between near states.

## VFA Prediction

The problem of *prediction* is the problem of estimating the value function $v^{\pi}(s)$ using the value function approximation $\hat{v}(s;\textbf{w})$. 

The objective for this task can be represented as a *Mean Squared Value Error* (MSVE) between the two:
$$ \mathrm{MSVE}(\mathbf{w}) = \mathbb{E}_{s \sim \mu^\pi}\!\left[ \big(v^\pi(s) - \hat{v}(s; \mathbf{w})\big)^2 \right] $$

To optimize this objective, we can progressively adapt the weights $\mathbf{w}$ using SGD:
$$ \mathbf{w} \leftarrow \mathbf{w} + \alpha \big(v^\pi(S_t) - \hat{v}(S_t; \mathbf{w})\big) \, \nabla_{\mathbf{w}} \hat{v}(S_t; \mathbf{w}) $$

Notice that $v^\pi(S_t)$ is obviously unknown. Therefore, we replace it with a *target*:
* **MC Prediction with FA** uses the target $G_t$.
* **TD Prediction with FA** uses the target $R_{t+1} + \gamma \cdot \hat{v}(S_{t+1}; \mathbf{w})$.


## VFA Control

The problem of control is the problem of finding an optimal policy $\pi^{\star}$ by estimating the optimal action-value function $q^{\star}(s,a)$ using a function approximator $\hat{q}(s,a;\mathbf{w})$.

Unlike prediction, where the policy $\pi$ is fixed and we estimate $v^\pi$, in control the policy is improved based on the current action-value estimates, using a greedy or $\epsilon$-greedy policy over $\hat{q}$.

The ideal objective would therefore be to minimize the error:
$$ \mathrm{MSQE}(\mathbf{w}) = \mathbb{E}_{(s, a) \sim \mu}\!\left[ \big(q^\star(s, a) - \hat{q}(s, a; \mathbf{w})\big)^2 \right] $$

As in prediction, we replace the unknown term $q^\star(s, a)$ with a *target*:
* **MC Prediction with FA** uses again the target $G_t$.
* **Q-Learning with FA** uses the target:
    $$ y_t = R_{t+1} + \gamma \cdot \max_a \hat{q}(S_{t+1}, a; \mathbf{w}) $$
* **SARSA with FA** uses the target:
    $$ y_t = R_{t+1} + \gamma \cdot \hat{q}(S_{t+1}, A_{t+1}; \mathbf{w}) $$


### VFA Semi-Gradient Q-Learning

Considering Q-Learning, we can write the loss function as:
$$ \mathcal{L}_t(\mathbf{w}) = \tfrac{1}{2} \big[ y_t - \hat{q}(S_t, A_t; \mathbf{w}) \big]^2 $$
which yields the Gradient Descente Update:
$$ \mathbf{w} \leftarrow \mathbf{w} - \alpha \, \nabla_{\mathbf{w}} \mathcal{L}_t(\mathbf{w})$$

The full gradient $\nabla_{\mathbf{w}} \mathcal{L}_t$ is:
$$ \nabla_{\mathbf{w}} \mathcal{L}_t = - \big[y_t - \hat{q}(S_t, A_t; \mathbf{w})\big] \, \big[ \nabla_{\mathbf{w}} \hat{q}(S_t, A_t; \mathbf{w}) - \gamma \nabla_{\mathbf{w}} \max_a \hat{q}(S_{t+1}, a; \mathbf{w}) \big] $$
however, VFA Control most often uses the *semi-gradient*, which drops the second term and simply keeps:
$$ \nabla_{\mathbf{w}} \mathcal{L}_t(\mathbf{w}) = - \big[y_t - \hat{q}(S_t, A_t; \mathbf{w})\big] \, \big[ \nabla_{\mathbf{w}} \hat{q}(S_t, A_t; \mathbf{w}) \big] $$
The update therefore becomes:
$$ \mathbf{w} \leftarrow \mathbf{w} + \alpha \big[y_t - \hat{q}(S_t, A_t; \mathbf{w})\big] \, \nabla_{\mathbf{w}} \hat{q}(S_t, A_t; \mathbf{w}) $$

This choice makes it so that:
* The (half) gradient which we are following through SGD comes solely from the *prediction*.
* The *target*, which also depends on $\mathbf{w}$, is handled as a constant (although it is not).

There are 3 reasons for this:
1. **Computational Simplicity**. As we avoid having to differentiate through the target, we simplify significantly the computation. 
In fact, we only need to:
    * Evaluate $\hat{q}(S_{t+1}, \cdot)$ to find the maximum $a$, for the TD target.
    * Compute the gradient only at $(S_t, A_t)$.

2. **Empirical Effectiveness**. Semi-gradient methods have been shown to perform well, despite using only a part of the gradient.

3. **Fidelity to the Bootstrap Idea**. The idea of bootstrapping is to fix a certain reference estimate for $S_{t+1}$ (given my $\hat{q}(S_{t+1}, \cdot)$) and use that to improve the estimate for $S_t$. Differentiating through the fixed reference estimate would therefore conflate with the idea that the target is something to reach for.

> Importantly, by dropping the second term of the gradient we are running a gradient descent on a minimum point that is different from the one that we would follow if we were to use the full gradient. In other words:
> * Convergence guarantees do not apply anymore.
> * The target moves as we change values for $\mathbf{w}$, which makes the training harder.

