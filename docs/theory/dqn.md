# 4. Deep Reinforcement Learning

> Type: **Deep Learning**
>
> State Space: Continuous, $\mathcal{S} = (x, y) \in [0, W] \times [0, H]$
>
> Action Space: Discrete $\mathcal{A} = \{ 0, 1, 2, 3 \} $

We have seen that, with **Function Approximation**, we can build a parametric function $\hat{q}_w$ to approximate the real Q-function, and we can train it to find the values for the weights $\mathbf{w}$ that better suit this approximation.

This definition also includes the possibility of fitting an entire **Deep Neural Network** inside the function $\hat{q}_w$. 
This choice, known as **Deep Reinforcement Learning**, faces the same instability challenges when applied with Off-Policy Algorithms.
We now try to understand the implications of **Deep RL** and some ways to contain instability.

## Implications of introducing a NN

Choosing a Deep Neural Networ as a parametric function has a couple of precise implications:

1. We move from **Feature Engineering** to **Representation Learning**. 
We do not need to define a feature map $\mathbf{x}(s)$ to extract the relevant aspects of the environment observation $s$. 
Instead, we can let the neural network know the relevant features by itself.

2. We need to **manually scale inputs** to the Neural Network.
In order to avoid magnitude issues with the Deep NN, we should manually scale each metric included in the state observation $s$ to a common fixed range.

Despite this cheap precisation, a naively implemented Q-Learning likely won't work. The general reason is once again the *deadly triad*, however, there are also two additional problems that involve Deep Q-Learning *specifically*.

## A) The Correlation Problem

Any proper Reinforcement Learning Algorithm assumes the data to come from an agent interacting with the environment, either physically or in a simulation.
As the agent interacts, it produces **trajectories** which can be used as experience for the learning process.

This procedure has the clear implication that **Training Examples generated from a Trajectory *ARE NOT* independent**. In fact, it is quite obvious that consecutive transitions in the trajectory will involve similar states, reached by similar actions.

This **High Correlation** has always been present in Reinforcement Learning, but only becomes a *real* issue when using Deep Neural Networks. In fact:
* **Tabular Q-Learning** is based on a convergence theory that does not require i.i.d. samples, therefore correlated trajectories are not a fundamental obstacle.
* **Linear VFA** follows the same description. Linear Semi-Gradient Q-Learning has explicitly been analyzed with samples coming from a Markov Trajectory and, thus, not i.i.d.
* **Deep VFA** also does not mathematically require i.i.d. samples *but*, as it usually is the case in machine learning, **correlation between samples typically causes optimization problems**.

### Solution: Experience Replay

The solution to the **Correlation Problem** is rather simple: 
1. Collect transitions from the trajectories generated interacting with the environment and store them into a **Replay Buffer**.
2. Sample random mini-batches from that buffer when training the Neural Network.

A hidden benefit of this approach is that we can actually re-play the same transition multiple times, if re-sampled. This means that we are leveraging each interaction with the environment better.

> Note that, in practice, the replay buffer has a fixed capacity and new experiences will typically take the place of older ones when it's full.

## B) Moving-Target Problem

The second problem has to do with moving targets.
We have already observed more than once how the application of **Gradient Descent** to Reinforcement Learning comes with the observation that there is no fixed, well-defined **label** to refer to at every epoch.
Instead, the **target** is built using our own estimate, which we are updating over time.
As a consequence, the learning targets are indeed *moving targets*.

Again, note that this is not something that has appeared just now:
* **Tabular Q-Learning** also uses moving targets, but does not have the property that causes targets to move indirectly due to closeby targets moving. This mathematically guarantees that training will converge, even with moving targets.
* **Linear VFA** has moving targets and also the possibility of divergence, as we have seen with the Baird's example for the deadly triad. However, in general, the moving-target problem is only rarely present.
* **Deep VFA** **reacts much worse to the Moving-Target Problem**. This is due to the parameters being more extensively tied together, thus amplifying the instability that only rarely affects Linear VFA.

### Solution: Target Network

The solution to the *Moving-Target Problem* is to reduce the oscillations by *fixing* a **Target** that the Network can follow.
Of course, the target cannot be fixed forever (unless we know the real value for the Q-function) or else we won't really be learning much.

The idea being applied in **Deep Q-Learning** (DQN) is therefore to have two networks:
1. The **Online Network** is the network that we update. It's the one we have always considered up until now.
    $$ Q(s, a; \mathbf{w}) $$
2. The **Target Network** is the one we use to build the TD Targets. It is held fixed for some number of training steps, after which it is updated with the newly learned weights of the Online Network.
    $$ Q(s, a; \mathbf{w}^-), \quad y_t = R_{t+1} + \gamma \max_a Q(S_{t+1}, a; \mathbf{w}^-) $$
That is, after $C$ updates we perform $\mathbf{w}^- \leftarrow \mathbf{w}$.

