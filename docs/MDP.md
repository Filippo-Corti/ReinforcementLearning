# MDP Formalization

This file contains the formalization of the MDP for the car racing problem.

## Simulation Details

The decision rate of the agent is set to $\Delta_{t_{agent}} = 0.04s$, meaning that the agent effectively interacts with the environment $25$ times per second. Note that the value of this variable conditions the minimum number of steps needed to finish a lap of a circuit, and therefore the value for the effective horizon $T_{\max}$.

Importantly, if we were to choose a large $\Delta_{t_{agent}}$, we would increase the chance of the car moving out of bounds between one agent-timestep and the next one.
For this reason, we define a simulation timestep of $\Delta_{t_{phys}} = 0.01s$. The simulation runs at this frequency, checking at each physical timestep if the car is colliding with an obstacle.
In other words, the agent only reacts every $4$ simulation steps, choosing the action to perform for the next $4$. 

## Environment State

$$ s_t = (x_t, y_t, \theta_t, v_t, C) $$

Where:
* $x_t, y_t, \theta_t$ describe the current pose of the Car (location and orientation).
* $v_t$ describes the current velocity of the car.
* $C$ is the configuration of the circuit, representing the environment.

This represents fully the environment but it is not what the agent observes.

## Observed State (1) - Frenet Coordinates

$$ o_t^{\text{Frenet}} = (d_t, \phi_{e,t}, v_t, \bar{\kappa}_t) $$

Where:
* $d_t$ is the lateral distance from the centerline of the track.
* $\phi_{e,t}$ is the heading error. That is, the discrepancy between car heading and track heading.
* $v_t$ is the current velocity.
* $\bar{\kappa}_t$ is a velocity-dependent summary of the track curvature ahead.

This is a richer, **Markov-like** observation of the environment. It assumes that the
car can localize itself on the track and has access to the circuit geometry. It is
not strictly Markov: two track locations can share the same local summary while
having different geometry farther ahead. The representation is nevertheless
intended to expose the information most relevant to short-horizon control without
giving the policy absolute Cartesian coordinates.

The local track geometry is preprocessed and stored with the track, as specified in [`TRACK.md`](TRACK.md).

## Observed State (2) - LiDAR Readings

$$ o_t^{\text{LiDAR}} = (v_t, R) $$

Where:
* $v_t$ describes current velocity.
* $R$ represents the readings from a LiDAR.

This is a lower-level observation, harder to interpret directly and which assumes that the robot does not know the full circuit.
This is a real **Partially Observable MDP**, as it is designed willingly to consider partial observability via local sensing and nothing more.

More precisely, we choose to model a LiDAR sensor with $16$ rays whose first and
last rays are included in a field of view (FOV) of $200°$. This corresponds to an
angular separation of $200°/(16-1) \approx 13.33°$. Range, normalization and
no-hit semantics are specified in [`TRACK.md`](TRACK.md).

## Action Space

$$ a_t = (a_t^{throttle}, a_t^{steer}) \in [-1,1]^2 $$

Where:
* $a_t^{throttle}$ represents acceleration ($a_t^{throttle}>0$) or braking ($a_t^{throttle}<0$).
* $a_t^{steer}$ represents steering, from left ($a_t^{steer}>0$) to right ($a_t^{steer}<0$).

In the environment, these normalized actions are mapped to physical controls as:

$$ \bar{a}_t^{throttle} = a_{max} \cdot a_t^{throttle}, \quad
 \bar{a}_t^{steer} = \delta_{max} \cdot a_t^{steer} $$

where:
* $a_{max} = 9.26 m/s^2$ is the maximum acceleration magnitude (corresponding to $0-100km/h$ in $3s$).
* $\delta_{max} = 30°$ is the maximum steering angle.

Reversing is not allowed, as assumed not necessary in an F1 circuit.

## Transition Kernel

Given the current environment state $s_t = (x_t, y_t, \theta_t, v_t, C)$ and a control action $a_t = (a_t^{throttle}, a_t^{steer})$, the transition to:
$$s_{t+1} = (x_{t+1}, y_{t+1}, \theta_{t+1}, v_{t+1}, C)$$
under deterministic constraints is assumed to be expressed by the **bicycle model** equations:

$$
\begin{aligned}
    \dot{x} = v_t \cos(\theta_t) \quad x_{t+1} = x_{t} + \Delta_t \dot{x} \\
    \dot{y} = v_t \sin(\theta_t) \quad y_{t+1} = y_{t} + \Delta_t \dot{y} \\
    \dot{\theta} = \frac{v_t}{L} \tan(\bar{a}_t^{steer}) \quad \theta_{t+1} = \theta_{t} + \Delta_t \dot{\theta} \\
    \dot{v} = \bar{a}_t^{throttle} \quad v_{t+1} = v_{t} + \Delta_t \dot{v}
\end{aligned}
$$

Where $L = 3.6m$ is the wheelbase (the distance between front and rear axles).
These equations are integrated with explicit Euler steps using
$\Delta_t=\Delta_{t_{phys}}=0.01s$. One agent action is held constant across four
successive physics steps, and collision is checked after each physics step.
Notably:
* Since reversing is not allowed, enforce $v_{t+1} \ge 0$.
* Also enforce $v_{t+1} \le v_{max}$, with $v_{max} \approx 70 m/s$ (around $250km/h$).

### Version 0 Physics Limitations

The first environment version deliberately uses the kinematic bicycle model
above, treats the car as a point at $(x_t,y_t)$ for collision detection, and does
not model lateral grip, aerodynamic drag, tire slip or steering-rate limits. As a
result, full throttle may remain optimal even in sharp corners. A later environment
version will add a grip constraint and a finite vehicle footprint; reward tuning
intended to produce braking behaviour must wait until that version. These are
model limitations, not behaviours to hide through reward shaping.

### Terminal States

A state $s_t$ is `terminal` if:
* $s_t \in \mathcal{F}$. That is, the car is at the finish line of the circuit. Or,
* $s_t \in \mathcal{W}$. That is, the car has just hit a wall (which are the boundaries of the track).

Indications on how to check if $s_t \in \mathcal{F}$ or $s_t \in \mathcal{W}$ are in [`TRACK.md`](TRACK.md).

In Gymnasium terms, finishing and crashing set `terminated=True`. Reaching
$T_{\max}$ without either event sets `truncated=True` and does not turn the state
into an MDP terminal state.

### Initial States

A simple choice would be to set the initial state to a fixed state where the car is right on the start line:
$$ s_0 = (x_0, y_0, \theta_0, 0, C) $$

For training, a better choice is typically a random location along the track, with some random velocities and headings. This increases exploration and training usually converges quicker.

At first, we consider the simple choice of the start line.

### Random Noise

If we later want to model a noisy environment, Gaussian noise can be added to the
four dynamic components:

$$ s_{t+1} = f(s_t, a_t) + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \Sigma) $$

## Reward Function

$$
r_t(s_t, s_{t+1}) =
\begin{cases}
-R_{\text{crash}} & \text{if } s_{t+1} \in \mathcal{W} \\
R_{\text{finish}} & \text{if } s_{t+1} \in \mathcal{F} \\
-c_{\text{step}} + c_{\text{prog}} \cdot \Delta\tilde{s}_t & \text{otherwise}
\end{cases}
$$

Collision is checked before finish so crossing the gate while off-track cannot
receive the finish reward.

Where:
* $R_{\text{finish}} = 100$
* $R_{\text{crash}} = 5$
* $c_{\text{step}}= \rho \cdot \Delta_{t_{agent}} = 0.02$, with $\rho = 0.5s^{-1}$ representing the cost over one agent step.
* $c_{\text{prog}}=100$
* $\Delta \tilde{s}_t$ is the progress term, computed as a normalized difference between the current and next locations (see [`TRACK.md`](TRACK.md)).

### Why these magnitudes

The coefficients are not free. They must preserve two orderings, or the task
becomes unlearnable by any policy-gradient method regardless of its update rule.

**Trying must beat doing nothing.** The car starts at $v_0=0$, and $v=0$ is an
absorbing state of the transition kernel: with zero speed the pose and heading
cannot change, so a policy with negative mean throttle receives a constant
observation and a constant reward until truncation. That stalled policy earns
$-c_{\text{step}}T_{\max}$. Any exploratory attempt to drive risks
$-R_{\text{crash}}$. If $R_{\text{crash}} > c_{\text{step}}T_{\max}$, the stall
strictly dominates every attempt, and the stall is exactly the behaviour that
gradient ascent can reach from a neutral initialization. The requirement is
therefore

$$ R_{\text{crash}} < c_{\text{step}} \cdot T_{\max}, $$

here $5 < 20$.

**Reaching further must be measurably better.** The shaped term is the only
signal that distinguishes a policy which crashes at 10% of the lap from one that
crashes at 90%. Its total authority over an episode is $c_{\text{prog}}$, against
a one-off $R_{\text{crash}}$. When $c_{\text{prog}} \ll R_{\text{crash}}$ the
return is, to within estimator noise, a function of *whether* the episode ended
in a crash and not of *where*, so there is no gradient toward driving further.
The requirement is

$$ c_{\text{prog}} \gg R_{\text{crash}}, $$

here $100 \gg 5$.

An earlier version of this document used $R_{\text{finish}}=10$,
$R_{\text{crash}}=20$, $\rho=0.05s^{-1}$ and $c_{\text{prog}}=1$. Both
inequalities ran the wrong way. Under those values a policy had to already
complete roughly two thirds of its laps before attempting the lap beat standing
still in expectation, which no algorithm can bootstrap from a $0\%$ completion
rate. REINFORCE, A2C and PPO all converged to the stall.

The maximum episode length is $T_{\max}=1000$ agent steps, corresponding to
$40s$. The training circuit is approximately one fifth of the original circuit
length, so its target lap is $18s$, or $18/0.04=450$ agent steps. The cap keeps
the same little-more-than-two-times safety margin used by the original
$5000$-step cap over a $2250$-step target lap. Both the circuit scale and cap are
configuration values and can be increased together for a longer task.

The signs and approximate undiscounted totals are:

* The step penalty over an $18s$ lap is
  $-c_{\text{step}}\times 18/\Delta_{t_{agent}}=-9$.
* The normalized progress accumulated over one forward lap is approximately
  $+100$.
* Including the finish reward, an $18s$ lap therefore returns approximately
  $100+100-9=191$. The exact value differs by at most one shaped transition
  because the terminal branch replaces the normal step reward.
* Remaining stationary until truncation returns
  $-c_{\text{step}}\times T_{\max}=-20$, the worst outcome available.
* An immediate crash returns $-R_{\text{crash}}=-5$.
* Crashing after one quarter of the lap returns approximately
  $100\times0.25-5-c_{\text{step}}\times T_{\text{crash}}\approx+18$, which is
  above both the stall and the immediate crash. The return is monotone in
  distance covered and decreasing in time taken, which is the intended
  shortest-time objective.

These reference values must be covered by reward tests when the environment is
implemented.

## Discounted Horizon Parameter

The learning contract fixes $\gamma=0.9995$ for all three algorithms. This value
is chosen from the task timescale:

* Its effective geometric horizon is $1/(1-\gamma)=2000$ agent steps, or $80s$.
* A reward received after an $18s$ lap is weighted by
  $\gamma^{450}\approx0.798$, rather than being effectively discarded.
* At the $1000$-step time limit, the discount weight is still approximately
  $0.606$.

Since the underlying objective is a finite-horizon shortest-time problem,
$\gamma=1$ would be a meaningful subject for a separate comparison. It is not a
factor in the approved experiments: changing $\gamma$ by algorithm or actor size
would confound the intended comparisons. Exact target and boundary semantics are
specified in [`LEARNING.md`](LEARNING.md).
