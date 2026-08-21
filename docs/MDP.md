# MDP Formalization

This file contains the formalization of the MDP for the car racing problem.

## Simulation Details

The simulation models time in seconds. 
Every $\Delta_{t_{phys}} = 0.01s$, the state of the agent is recomputed according to its current controls and the environments checks for any collision with obstacles.

The decision rate of the agent is set to $\Delta_{t_{agent}} = 0.04s$, meaning that the agent effectively interacts with the environment $25$ times per second. 
In other words, the agent only reacts every $4$ simulation steps, choosing the action to perform for the next $4$. 

> Note that the value of $\Delta_{t_{agent}}$ conditions the minimum number of steps needed to finish a lap of a circuit, and therefore the value for the effective horizon $T_{\max}$.

## Environment State

$$ s_t = (x_t, y_t, \theta_t, v_t, \delta_t, C) $$

Where:
* $x_t, y_t, \theta_t$ describe the current pose of the car (location and orientation).
* $v_t, \delta_t$ describe the current internal state of the car (velocity and front-wheel angle).
* $C$ is the configuration of the circuit, representing the environment. For more information on that, check [`TRACK.md`](TRACK.md).

This represents fully the environment but it is not what the agent observes.

> Note that $v_t$ and $\delta_t$ are part of the environment because they do not correspond *directly* to the controls of the agent. In fact, the agent controls acceleration and steering, but the actual velocity and heading are subject to physical forces.
> If this were the case, there would be no need to have them as part of the (observed) state. 

## Observed State (1) - Frenet Coordinates

A rich **Markov-like** observation of the environment is the one provided by the **Frenet Coordinates**.
This observation assume that the car can localize itself on the track and has access to the circuit geometry.

$$ o_t^{\text{Frenet}} = (d_t, \phi_{e,t}, v_t, \delta_t, \bar{\kappa}_t) $$

Where:
* $d_t$ is the lateral distance from the centerline of the track.
* $\phi_{e,t}$ is the heading error. That is, the discrepancy between car heading and track heading.
* $v_t$ is the current velocity.
* $\delta_t$ is the current front-wheel steering angle.
* $\bar{\kappa}_t$ is a velocity-dependent summary of the track curvature ahead.

> Note that the observation is not strictly **Markov**: for example, the car could be in two different tracks that share a very similar section. They would be locally equal, but as the car progresses the control action would have different effects.
> Nevertheless, the representation is intended to expose the most relevant information to short-horizon.

## Observed State (2) - LiDAR Readings

A lower-level observation is given by the **LiDAR Readings**.
This observation assumes that the car does not know the full circuit, and can only use a LiDAR rangefinder sensor to observe the track around itself.

$$ o_t^{\text{LiDAR}} = (v_t, \delta_t, R) $$

Where:
* $v_t$ describes current velocity.
* $\delta_t$ describes current front-wheel steering angle.
* $R$ represents the readings from a LiDAR.

More precisely, we choose to model a LiDAR sensor with $16$ rays whose first and last rays are included in a field of view (FOV) of $200°$.
This corresponds to an angular separation of $200°/(16-1) \approx 13.33°$. 

> This is a real **Partially Observable MDP**, as it is willingly designed to consider partial observability via local sensing, and nothing more than that.


## Action Space

The agent interacts with the environment via control actions $a_t$:

$$ a_t = (a_t^{throttle}, a_t^{steer}) \in [-1,1]^2 $$

Where:
* $a_t^{throttle}$ represents acceleration ($a_t^{throttle}>0$) or braking ($a_t^{throttle}<0$).
* $a_t^{steer}$ represents steering, from left ($a_t^{steer}>0$) to right ($a_t^{steer}<0$).

These are normalized actions, which are mapped to physical controls as:

$$
\bar{a}_t^{throttle} =
\begin{cases}
a_{max}\cdot a_t^{throttle}, & a_t^{throttle} \ge 0,\\
b_{max}\cdot a_t^{throttle}, & a_t^{throttle} < 0,
\end{cases}
\qquad
\delta_t^{\star} = \delta_{max} \cdot a_t^{steer}
$$

where:
* $a_{max} = 9.26 m/s^2$ is the engine-limited forward acceleration (corresponding to $0-100km/h$ in $3s$).
* $b_{max} = 20 m/s^2$ is the grip-limited braking deceleration.
* $\delta_{max} = 30°$ is the maximum steering angle.

Acceleration and braking differ, as with real cars: acceleration is limited by the engine, while braking is limited only by tires.
The effects of these physical controls are not straightforward.
More on this is specified below.

> Note that reversing is not allowed, as assumed not necessary in an F1 circuit.

## Transition Kernel

Given the current environment state $s_t = (x_t, y_t, \theta_t, v_t, \delta_t, C)$ and a control action $a_t = (a_t^{throttle}, a_t^{steer})$, the transition to:
$$s_{t+1} = (x_{t+1}, y_{t+1}, \theta_{t+1}, v_{t+1}, \delta_{t+1}, C)$$
under deterministic constraints is assumed to be expressed by the **bicycle model** equations:

$$
\begin{aligned}
    \dot{x} = v_t \cos(\theta_t) \quad &x_{t+1} = x_{t} + \Delta_t \dot{x} \\
    \dot{y} = v_t \sin(\theta_t) \quad &y_{t+1} = y_{t} + \Delta_t \dot{y} \\
    \dot{\theta} = \frac{v_t}{L} \tan(\tilde{\delta}_{t+1}) \quad &\theta_{t+1} = \theta_{t} + \Delta_t \dot{\theta} \\
    \dot{v} = \bar{a}_t^{throttle} - c_d v_t^2 \quad &v_{t+1} = v_{t} + \Delta_t \dot{v}
\end{aligned}
$$

Where $L = 3.6m$ is the wheelbase (the distance between front and rear axles).
Some additional constraints are:
* Since reversing is not allowed, enforce $v_{t+1} \ge 0$.
* Also enforce $v_{t+1} \le v_{max}$, with $v_{max} = 70 m/s$ (around $250km/h$).

The environment models the effects of the physical control action according to three physical principles.

#### Aerodynamic Drag

The term $c_d v_t^2$ opposes motion at all times, by reducing the acceleration desired by the car. 
The constant $c_d$ is derived so that $v_{max}$ is exactly the speed at which drag cancels full throttle, making it impossible for the car to accelerate further:

$$ c_d = \frac{a_{max}}{v_{max}^2} \approx 1.89\cdot 10^{-3}\,\mathrm{m^{-1}} $$

#### Steering Rate Limit

The wheels move toward the requested angle at no more than $\dot\delta_{max} = 180°/s$:

$$ \delta_{t+1} = \delta_t + \mathrm{clip}\left(\delta_t^{\star} - \delta_t,\;
   -\dot\delta_{max}\Delta_t,\; +\dot\delta_{max}\Delta_t\right) $$

This means that a full sweep from full-right steer to full-left steer takes a third of a second (about
eight agent steps), instead of being available instantaneously. 

#### Tire friction

The car decides the wheels' orientation.
However, when cruising at high speeds, it is possible that the car won't follow the wheels and instead **understeer**.
This idea is modelled through tire friction.

Longitudinal and lateral tie forces share one budget $\mu g = 20\,\mathrm{m/s^2}$, approximately $2g$. 
Writing the lateral demand of the rate-limited angle as:

$$ a^{lat}_{t+1} = \frac{v_t^2 \tan(\delta_{t+1})}{L}, $$

the lateral acceleration the tires can still deliver is whatever the friction circle leaves after the longitudinal demand:

$$ a^{lat}_{avail} = \sqrt{\max\left(0,\;(\mu g)^2 - \min(|\bar{a}_t^{throttle}|, \mu g)^2\right)} $$

and the angle that actually steers the car is reduced to fit it:

$$
\tilde{\delta}_{t+1} =
\begin{cases}
\delta_{t+1}, & |a^{lat}_{t+1}| \le a^{lat}_{avail},\\[4pt]
\mathrm{sign}(\delta_{t+1})\arctan\left(\dfrac{a^{lat}_{avail}\,L}{v_t^2}\right), & \text{otherwise.}
\end{cases}
$$

> At $v = 70\,\mathrm{m/s}$ the tightest corner of a typical generated circuit ($R \approx 15\,\mathrm m$) would demand $330\,\mathrm{m/s^2}$, about $34g$; 
> the friction budget caps the car at $17\,\mathrm{m/s}$ through that corner while straights remain available at $v_{max}$. 

#### Remaining physics limitations

* The model still treats the car as a point at $(x_t,y_t)$ for collision detection.
* The car has no tire slip angle, load transfer, or speed-dependent downforce.
* The friction budget is a constant rather than growing with speed as it does on a real downforce car.

### Terminal States

A state $s_t$ is `terminal` if:
* $s_t \in \mathcal{F}$. That is, the car is at the finish line of the circuit. Or,
* $s_t \in \mathcal{W}$. That is, the car has just hit a wall (which are the boundaries of the track). Or,
* $s_t \in \mathcal{S}$. That is, the car has stopped racing and is simply stalling.

Finishing, crashing and stalling set `terminated=True`.
Reaching $T_{\max}$ without any of them sets `truncated=True` and does not turn the state into an MDP terminal state.

Indications on how to check if $s_t \in \mathcal{F}$ or $s_t \in \mathcal{W}$ are in [`TRACK.md`](TRACK.md).

Regarding the stall set $\mathcal{S}$: a car that has advanced less than $\sigma_{\min}=1\,\mathrm m$ over the last $T_{\text{stall}}=3\,\mathrm s$ has stopped racing. Since the state won't change, the control actions also won't change (except for the noise), so the episode ends early. 

> Note that a stalled episode is still charged a negative reward for the entire time that it would have spent idling to the limit.
> In other words, this is more of a simulation shortcut than an actual modeling choice.

### Initial States

#### Training

The start pose in training is sampled uniformly over the circuit:

$$
s_0 \sim \left(
  s^{arc} \sim \mathcal{U}[0, L_{track}),\;
  d_0 \sim \mathcal{U}[-0.5\tfrac{W}{2}, 0.5\tfrac{W}{2}],\;
  \phi_{e,0} \sim \mathcal{U}[-0.2, 0.2],\;
  v_0 \sim \mathcal{U}[0, 0.15\,v_{max}],\;
  \delta_0 = 0
\right)
$$

The finish gate moves with the start, so a lap is always one full circuit from wherever the car was placed.

> A fixed zero-speed start makes successful launch a prerequisite for collecting experience from the remainder of the circuit.
> This could be an issue if the agent ends up (or starts) in a spot where the throttle is non-positive: the car won't move ($v=0$) and the agent won't explore any other part of the circuit.
> Sampling the pose solves this. 

#### Evaluation

Deterministic evaluation always launches from the canonical start line with $v_0 = 0$:

$$ s_0 = (x_0, y_0, \theta_0, 0, 0, C) $$


## Reward Function

The reward function for the MDP is the following:

$$
r_t(s_t, s_{t+1}) =
\begin{cases}
-R_{\text{crash}} & \text{if } s_{t+1} \in \mathcal{W} \\
R_{\text{finish}} + R_{\text{lap}}\left(1 - \dfrac{t}{T_{\max}}\right) & \text{if } s_{t+1} \in \mathcal{F} \\
-c_{\text{step}}\left(T_{\max} - t + 1\right) & \text{if } s_{t+1} \in \mathcal{S} \\
-c_{\text{step}} + c_{\text{prog}} \cdot \Delta\tilde{s}_t & \text{otherwise}
\end{cases}
$$

Where:
* $R_{\text{finish}} = 100$
* $R_{\text{lap}} = 140$ is the completion reward, scaled by lap time.
* $R_{\text{crash}} = 5$
* $c_{\text{step}}= \rho \cdot \Delta_{t_{agent}} = 0.04$, with $\rho = 1s^{-1}$ representing the cost over one agent step.
* $c_{\text{prog}}=100$
* $\Delta \tilde{s}_t$ is the progress term, computed as a normalized difference between the current and next locations (see [`TRACK.md`](TRACK.md)).

### Choosing the coefficients

The coefficients cannot be chosen arbitrarily.
In order for the task to be properly learnable, three conditions are imposed.

1. **Trying must beat doing nothing.** A policy that stalls throughout the entirety of the episode (that is, for $T_{\max}$ steps) should be punished harder than one that crashes, even one that crashes immediately. The requirement is:
  $$ R_{\text{crash}} < c_{\text{step}} \cdot T_{\max}, $$

2. **Reaching further must be measurably better.** The progress term is the only signal that distinguishes a policy which crashes immediately from one that crashes at 90% of the lap. The requirement is that the progress term should prevail over the crashes:
  $$ c_{\text{prog}} \gg R_{\text{crash}}, $$

3. **A fast lap must beat a slow lap by a visible margin.** If the goal is racing, the car should not just be incentivized for progressing, but also for finishing fast. Increasing $\rho$ does not work, as it leads to policies that favour instant crashing over finishing. The solution is to add a proper bonus $R_{\text{lap}}$, that scales with the unused episode clock:

  $$ \frac{r(\text{fast lap}) - r(\text{slow lap})}{r(\text{fast lap})} > 0.2, $$

With the values specified above, all 3 conditions are satisfied:
1. $5 < 40$
2. $100 \gg 5$
3. Considering $T_{\max}=1000$, a deterministic reference controller takes at best $22s$ to finish a lap (empirically, over a set of random circuits). Assuming a slow lap to take $1.5 \times$ that, a slow lap takes $33s$. Condition 3 then solves as:
$$
r_{\text{fast}}
= R_{\text{finish}} + R_{\text{lap}}\left(1-\frac{22}{40}\right) - 22
= 100 + 140(1-0.55)-22
= 141,\\
r_{\text{slow}}
= R_{\text{finish}} + R_{\text{lap}}\left(1-\frac{33}{40}\right) - 33
= 100 + 140(1-0.825)-33
= 91.5,\\
\frac{r_{\text{fast}}-r_{\text{slow}}}{r_{\text{fast}}}
= \frac{141-91.5}{141}
\approx 0.351 > 0.2.
$$

### Discounted Horizon Parameter

The experimental setting has a finite horizon.
To align with this decision, no discount factor $\gamma$ is applied:
$$ \gamma = 1 $$

A different choice for $\gamma$ would model as an incentive to progress quickly at the start of an episode.
The algorithm would still learn and results would be comparably better, but it would compromise the theoretical clarity of the finite horizon objective.