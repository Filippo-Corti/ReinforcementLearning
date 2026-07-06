# RL Car Racing

## Project Proposal (Project ID: PG-4):

### Main Focus
Policy gradient, deep neural policies

### Scientific Objective 
Understand the impact of the complexity of the policy space on the performance of the agent and the amount of interactions needed to converge.

### Problem Description 
You want to control an autonomous Formula 1 car so that it completes a circuit in the shortest time possible. 
The car has access to its position (relative to the circuit) and velocity. 
It controls the acceleration and the steering wheel. It should be heavily penalized for going off track.

### Tasks
1. Choose/design a circuit (e.g. The Circuit de Monaco) and model the problem as an MDP with continuous actions. Think carefully about how to model the circuit and the relative position of the car. You are allowed to include in the state additional information about the specific circuit you are trying to solve, such as landmarks or information about the curvature, if you think this is helpful.
2. Define a reward function that encourages the agent to complete the circuit in the shortest time possible without going off track.
3. Define a parametric policy (Gaussian or deterministic) that maps states to actions (or mean actions) using a deep neural network (e.g. a fully connected neural network a.k.a. multi-layer perceptron). Implement it so that you can easily try neural networks of different sizes (number of layers and width of the layers).
4. Train your agent using a deep RL algorithm of your choice and compare the results obtained with policy networks of different sizes, in terms of:
    - Final performance
    - Number of training episodes needed to converge
    - Time needed to converge (this is machine dependent so make sure to run all experiments on the same computer)

### Challenging Variants 
Try to learn a policy that can solve multiple circuits, in particular circuits not seen during training. You may want to train it on multiple, diverse circuits.


## MDP Formalization

### Environment State

$$ s_t = (x_t, y_t, \theta_t, v_t, C) $$

Where:
* $x_t, y_t, \theta_t$ describe the current pose of the Car (location and orientation).
* $v_t$ describse the current velocity of the car.
* $C$ is the configuration of the circuit, representing the environment.

This represents fully the environment but it is not what the agent observes.

### Observed State (1) - Frenet Coordinates

$$ o_t = (d_t, \phi_{e,t}, v_t, \kappa_t) $$

Where:
* $d_t$ is the lateral distance from the centerline of the track.
* $\phi_{e,t}$ is the heading error. That is, the discrepancy between car heading and track heading.
* $v_t$ is the current velocity.
* $\kappa_t$ is the change of the curvature of the road straight ahead.

This is a more rich observation of the environment, which assumes that the robot has a way to localize themselves on the track and has a full knowledge of the circuit.
It is more similar to a **Fully Observable MDP**, as although being a function of the real MDP state it is designed to be a Markov State representation.

### Observed State (2) - LiDAR Readings

$$ o_t = (v_t, R) $$

Where:
* $v_t$ describes current velocity.
* $R$ represents the readings from a LiDAR.

This is a lower-level observation, harder to interpret directly and which assumes that the robot does not know the full circuit.
This is a real **Partially Observable MDP**, as it is designed willingly to consider partial observability via local sensing and nothing more.

### Action Space

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

### Transition Kernel

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

Where $L$ is the wheelbase (the distance between fron and rear axles) and $\Delta_t$ is the discrete time step between one action and the next one. Notably:
* Since reversing is not allowed, enforce $v_{t+1} \ge 0$.
* Also enforce $v_{t+1} \le v_{max}$, with $v_{max} \approx 70 m/s$ (around $250km/h$).

#### Terminal States

A state $s_t$ is `terminal` if:
* $s_t \in \mathcal{F}$. That is, the car is at the finish line of the circuit.
* $s_t \in \mathcal{W}$. That is, the car has just hit a wall (which are the boundaries of the track).

#### Initial States

A simple choice would be to set the initial state to a fixed state where the car is right on the start line:
$$ s_0 = (x_0, y_0, \theta_0, 0, C) $$

For training, a better choice is typically a random location along the track, with some random velocities and headings. This increases exploration and training usually converges quicker.

#### Random Noise

If we want to model a noisy environment, we can simply add gaussian noise to the four component:

$$ s_{t+1} = f(s_t, a_t) + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \Sigma) $$

### Reward Function

$$
r_t =
\begin{cases}
R_{\text{finish}} & \text{if } s_{t+1} \in \mathcal{F} \\
-R_{\text{crash}} & \text{if } s_{t+1} \in \mathcal{W} \\
-c_{\text{step}} + c_{\text{prog}} \cdot \Delta\tilde{s}_t & \text{otherwise}
\end{cases}
$$

Where:
* $R_{\text{finish}} = R_{\text{crash}} = 10$
* $c_{\text{step}}=0.01$
* $c_{\text{prog}}=1$

Under these conditions, a reasonable choice for the maximum length of an episode is $T_{max} = 500$ steps. This was, in fact we guarantee that $0.01 \times 500 = 5$ is significantly worse than a slow, complete lap (which would be something like $0.01 \times 350 + 10 = 13.5$). 

### Discounted Horizon Parameters

A reasonable choice can be $\gamma = 0.99$ and $T_{max} = 500$.


## Frenet Coordinates and Track Progress

TODOs:
- How do we define the track?
- How do we determine the Frenet state given the current state?
- How do we compute progress reward?

# Future TODOs:

- If car goes always full throttle -> Add a lateral acceleration representing centripetal force. It is quadratically proportional to velocity, meaning that at high speed it is stronger. Maybe it should also influence steering effect? Or just move the car. Not sure yet.

- If training is unstable -> Add the previous action to the state observation

- If training is slow -> random starting distribution