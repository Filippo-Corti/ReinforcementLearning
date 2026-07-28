# MDP Formalization

This file contains the formalization of the MDP for the car racing problem.

## Simulation Details

The decision rate of the agent is set to $\Delta_{t_{agent}} = 0.04s$, meaning that the agent effectively interacts with the environment $25$ times per second. Note that the value of this variable conditions the minimum number of steps needed to finish a lab of a circuit, and therefore the value for the effective horizon $T_max$.

Importantly, if were to choose a large $\Delta_{t_{agent}}$, we would increase the chance of the car moving out of bounds between one agent-timestep and the next one. 
For this reason, we define a simulation timestep of $\Delta_{t_{phys}} = 0.01s$. The simulation runs at this frequency, checking at each physical timestep if the car is colliding with an obstacle.
In other words, the agent only reacts every $4$ simulation steps, choosing the action to perform for the next $4$. 

## Environment State

$$ s_t = (x_t, y_t, \theta_t, v_t, C) $$

Where:
* $x_t, y_t, \theta_t$ describe the current pose of the Car (location and orientation).
* $v_t$ describse the current velocity of the car.
* $C$ is the configuration of the circuit, representing the environment.

This represents fully the environment but it is not what the agent observes.

## Observed State (1) - Frenet Coordinates

$$ o_t^{\text{Frenet}} = (d_t, \phi_{e,t}, v_t, \kappa_t) $$

Where:
* $d_t$ is the lateral distance from the centerline of the track.
* $\phi_{e,t}$ is the heading error. That is, the discrepancy between car heading and track heading.
* $v_t$ is the current velocity.
* $\kappa_t$ is the change of the curvature of the road straight ahead.

This is a more rich observation of the environment, which assumes that the robot has a way to localize themselves on the track and has a full knowledge of the circuit.
It is more similar to a **Fully Observable MDP**, as despite being a function of the real MDP state it is designed to be a Markov State representation.

## Observed State (2) - LiDAR Readings

$$ o_t^{\text{LiDAR}} = (v_t, R) $$

Where:
* $v_t$ describes current velocity.
* $R$ represents the readings from a LiDAR.

This is a lower-level observation, harder to interpret directly and which assumes that the robot does not know the full circuit.
This is a real **Partially Observable MDP**, as it is designed willingly to consider partial observability via local sensing and nothing more.

More precisely, we choose to model a LiDAR sensor with $16$ rays, across a field of view (FOV) of $200$ degrees. This corresponds to one raycast every $12.5°$.

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

Where $L = 3.6m$ is the wheelbase (the distance between fron and rear axles) and $\Delta_t$ is the discrete time step between one action and the next one. Notably:
* Since reversing is not allowed, enforce $v_{t+1} \ge 0$.
* Also enforce $v_{t+1} \le v_{max}$, with $v_{max} \approx 70 m/s$ (around $250km/h$).

### Terminal States

A state $s_t$ is `terminal` if:
* $s_t \in \mathcal{F}$. That is, the car is at the finish line of the circuit. Or,
* $s_t \in \mathcal{W}$. That is, the car has just hit a wall (which are the boundaries of the track).

Indications on how to check if $s_t \in \mathcal{F}$ or $s_t \in \mathcal{W}$ are in [`TRACK.md`](TRACK.md).

### Initial States

A simple choice would be to set the initial state to a fixed state where the car is right on the start line:
$$ s_0 = (x_0, y_0, \theta_0, 0, C) $$

For training, a better choice is typically a random location along the track, with some random velocities and headings. This increases exploration and training usually converges quicker.

At first, we consider the simple choice of the start line.

### Random Noise

If we want to model a noisy environment, we can simply add gaussian noise to the four component:

$$ s_{t+1} = f(s_t, a_t) + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \Sigma) $$

## Reward Function

$$
r_t(s_t, s_{t+1}) =
\begin{cases}
R_{\text{finish}} & \text{if } s_{t+1} \in \mathcal{F} \\
-R_{\text{crash}} & \text{if } s_{t+1} \in \mathcal{W} \\
-c_{\text{step}} + c_{\text{prog}} \cdot \Delta\tilde{s}_t & \text{otherwise}
\end{cases}
$$

Where:
* $R_{\text{finish}} = 10$
* $R_{\text{crash}} = 20$
* $c_{\text{step}}= \rho \cdot \Delta_{t_{agent}} = 0.002$, with $\rho = 0.05s^{-1}$ simply representing the cost over a simulation step.
* $c_{\text{prog}}=1$
* $\Delta \tilde{s}_t$ is the progress term, computed as a normalized difference between the current and next locations (see [`TRACK.md`](TRACK.md)).

Under these conditions, a reasonable choice for the maximum length of an episode is $T_{max} = 5000$ steps. 
This way we can model a circuit that can be completed in around $90s$ with a fast but realistic lap (to verify this: $90/0.04 * 2.5$, with $2.5$ used to give a margin of learning to the agent during training). 
Moreover:
* A full, fast lap is finished with a reward of $c_{\text{step}} \times 90/\Delta_{t_{agent}} = 4.5$, which should make for a clear reward compared to an unfinished run.
* Avoiding crashes for the full horizon results in a reward of $c_{\text{step}} \times T_{max} = 10$, which should make for a clear reward compared to an instant crash.

## Discounted Horizon Parameters

A reasonable choice can be $\gamma = 0.99$ and $T_{max} = 500$.
