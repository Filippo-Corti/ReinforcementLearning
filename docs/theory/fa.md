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

In general, the idea is to replace the Q-table with a *parametric function*:
$$ 
\mathcal{Q} : \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}
$$ 
We therefore choose a family of functions $\hat{q}(s,a;\textbf{w})$, parametric by $\textbf{w} \in \mathbb{R}^{d}$, and we learn the value of $\textbf{w}$ so that $\hat{q}_{\textbf{w}}$ better approximates the real, unknown, $\mathcal{Q}$ function.

> The biggest advantage is *generalization*. By updating $\hat{q}_{\textbf{w}}$ in a certain point $(s, a)$ (a certain state), we also shape the function for the points near $(s, a)$. 
> We are making explicit use of the *similarity* between near states.
