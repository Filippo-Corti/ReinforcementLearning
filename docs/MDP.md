# MDP Formalization

This file contains the formalization of the MDP for the car racing problem.

## Simulation Details

The decision rate of the agent is set to $\Delta_{t_{agent}} = 0.04s$, meaning that the agent effectively interacts with the environment $25$ times per second. Note that the value of this variable conditions the minimum number of steps needed to finish a lap of a circuit, and therefore the value for the effective horizon $T_{\max}$.

Importantly, if we were to choose a large $\Delta_{t_{agent}}$, we would increase the chance of the car moving out of bounds between one agent-timestep and the next one.
For this reason, we define a simulation timestep of $\Delta_{t_{phys}} = 0.01s$. The simulation runs at this frequency, checking at each physical timestep if the car is colliding with an obstacle.
In other words, the agent only reacts every $4$ simulation steps, choosing the action to perform for the next $4$. 

## Environment State

$$ s_t = (x_t, y_t, \theta_t, v_t, \delta_t, C) $$

Where:
* $x_t, y_t, \theta_t$ describe the current pose of the Car (location and orientation).
* $v_t$ describes the current velocity of the car.
* $\delta_t$ is the current front-wheel steering angle.
* $C$ is the configuration of the circuit, representing the environment.

The steering angle is part of the state because it is rate limited: an action
requests an angle and the wheels travel toward it over several physics substeps,
so the angle in force at $t+1$ depends on the angle at $t$.
This is the same reason why the velocity is part of the state: the throttle does 
not tell how fast the car is moving.

This represents fully the environment but it is not what the agent observes.

## Observed State (1) - Frenet Coordinates

$$ o_t^{\text{Frenet}} = (d_t, \phi_{e,t}, v_t, \delta_t, \bar{\kappa}_t) $$

Where:
* $d_t$ is the lateral distance from the centerline of the track.
* $\phi_{e,t}$ is the heading error. That is, the discrepancy between car heading and track heading.
* $v_t$ is the current velocity.
* $\delta_t$ is the current front-wheel steering angle.
* $\bar{\kappa}_t$ is a velocity-dependent summary of the track curvature ahead.

This is a richer, **Markov-like** observation of the environment. It assumes that the
car can localize itself on the track and has access to the circuit geometry. It is
not strictly Markov: two track locations can share the same local summary while
having different geometry farther ahead. The representation is nevertheless
intended to expose the information most relevant to short-horizon control without
giving the policy absolute Cartesian coordinates.

The local track geometry is preprocessed and stored with the track, as specified in [`TRACK.md`](TRACK.md).

## Observed State (2) - LiDAR Readings

$$ o_t^{\text{LiDAR}} = (v_t, \delta_t, R) $$

Where:
* $v_t$ describes current velocity.
* $\delta_t$ describes current front-wheel steering angle.
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

Acceleration and braking differ, as it happens in real cars.
This is due to being limited by different things: the
engine cannot deliver more than $a_{max}$, while the brakes are limited only by
the tyres and therefore reach the full friction budget $\mu g$ defined below.

The steering component is a **requested** angle $\delta_t^{\star}$, not the angle
that takes effect. The action is a demand; the transition kernel decides what the
car actually does with it.
Again, this matches a real car: steering at eccessive speed causes underrotations.

Reversing is not allowed, as assumed not necessary in an F1 circuit.

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
These equations are integrated with explicit Euler steps using
$\Delta_t=\Delta_{t_{phys}}=0.01s$. One agent action is held constant across four
successive physics steps, and collision is checked after each physics step.
Notably:
* Since reversing is not allowed, enforce $v_{t+1} \ge 0$.
* Also enforce $v_{t+1} \le v_{max}$, with $v_{max} = 70 m/s$ (around $250km/h$).

Three limits stand between the requested action and the motion above.

**Aerodynamic drag.** The term $c_d v_t^2$ opposes motion at all times. Rather
than introduce a free constant, $c_d$ is derived so that $v_{max}$ is exactly the
speed at which drag cancels full throttle:

$$ c_d = \frac{a_{max}}{v_{max}^2} \approx 1.89\cdot 10^{-3}\,\mathrm{m^{-1}} $$

so the speed ceiling is a physical consequence rather than an imposed clamp. The
clamp at $v_{max}$ is retained only as a numerical guard.

**Steering rate limit.** The wheels move toward the requested angle at no more
than $\dot\delta_{max} = 180°/s$:

$$ \delta_{t+1} = \delta_t + \mathrm{clip}\left(\delta_t^{\star} - \delta_t,\;
   -\dot\delta_{max}\Delta_t,\; +\dot\delta_{max}\Delta_t\right) $$

A full sweep from lock to lock therefore takes a third of a second, or about
eight agent steps, instead of being available instantaneously. This is what makes
$\delta_t$ a state variable.

**Tyre friction budget.** Longitudinal and lateral tyre forces share one budget
$\mu g = 20\,\mathrm{m/s^2}$, approximately $2g$. Writing the lateral demand of
the rate-limited angle as

$$ a^{lat}_{t+1} = \frac{v_t^2 \tan(\delta_{t+1})}{L}, $$

the lateral acceleration the tyres can still deliver is whatever the friction
circle leaves after the longitudinal demand:

$$ a^{lat}_{avail} = \sqrt{\max\left(0,\;(\mu g)^2 - \min(|\bar{a}_t^{throttle}|, \mu g)^2\right)} $$

and the angle that actually steers the car is reduced to fit it:

$$
\tilde{\delta}_{t+1} =
\begin{cases}
\delta_{t+1}, & |a^{lat}_{t+1}| \le a^{lat}_{avail},\\[4pt]
\mathrm{sign}(\delta_{t+1})\arctan\left(\dfrac{a^{lat}_{avail}\,L}{v_t^2}\right), & \text{otherwise.}
\end{cases}
$$

The wheels stay where the driver put them; the car simply does not follow. This
is **understeer**: entering a corner too fast makes the car run wide and hit the
outer boundary, so the crash arises from geometry rather than from a rule that
declares excessive cornering illegal. Braking at the limit spends the whole
budget and leaves nothing to turn with, which is the trade the driver has to
plan around.

At $v = 70\,\mathrm{m/s}$ the tightest corner of a typical generated circuit
($R \approx 15\,\mathrm m$) would demand $330\,\mathrm{m/s^2}$, about $34g$;
the friction budget caps the car at $17\,\mathrm{m/s}$ through that corner while
straights remain available at $v_{max}$. Speed is therefore something the policy
has to spend and recover rather than simply hold.

### Remaining physics limitations

The model still treats the car as a point at $(x_t,y_t)$ for collision detection
and has no tyre slip angle, load transfer, or speed-dependent downforce: the
friction budget is a constant rather than growing with speed as it does on a real
downforce car. Cornering is quasi-static, so the model cannot represent a car
that is sliding. These are model limitations, not behaviours to hide through
reward shaping.

### Terminal States

A state $s_t$ is `terminal` if:
* $s_t \in \mathcal{F}$. That is, the car is at the finish line of the circuit. Or,
* $s_t \in \mathcal{W}$. That is, the car has just hit a wall (which are the boundaries of the track).
* $s_t \in \mathcal{S}$. That is, the car has stopped racing (see below).

Indications on how to check if $s_t \in \mathcal{F}$ or $s_t \in \mathcal{W}$ are in [`TRACK.md`](TRACK.md).

In Gymnasium terms, finishing, crashing and stalling set `terminated=True`.
Reaching $T_{\max}$ without any of them sets `truncated=True` and does not turn
the state into an MDP terminal state.

**The stall set $\mathcal{S}$.** A car that has advanced less than
$\sigma_{\min}=1\,\mathrm m$ over the last $T_{\text{stall}}=3\,\mathrm s$ has
stopped racing. Simulating it for the remaining episode produces no information,
so the episode ends there.

This must not become a cheap way out. A stalled episode is charged the entire
time penalty it would have paid by idling to the limit:

$$ r_t = -c_{\text{step}}\left(T_{\max} - t + 1\right)
   \qquad \text{on entering } \mathcal{S}, $$

so the return of standing still is identical to the return of standing still for
the full episode. Ending early saves the simulator, not the agent.

### Initial States

**Training.** The start pose is sampled uniformly over the circuit:

$$
s_0 \sim \left(
  s^{arc} \sim \mathcal{U}[0, L_{track}),\;
  d_0 \sim \mathcal{U}[-0.5\tfrac{W}{2}, 0.5\tfrac{W}{2}],\;
  \phi_{e,0} \sim \mathcal{U}[-0.2, 0.2],\;
  v_0 \sim \mathcal{U}[0, 0.15\,v_{max}],\;
  \delta_0 = 0
\right)
$$

A single fixed start makes the zero-speed launch the only state the agent ever
has to solve, and puts the whole rest of the circuit behind that one bottleneck.
It also interacts badly with the transition kernel: at $v=0$ the pose and heading
cannot change, so a deterministic policy whose mean throttle is non-positive at
that one state never moves at all, however well it drives everywhere else.
Sampling the pose removes both problems.

The finish gate moves with the start (see [`TRACK.md`](TRACK.md)), so a lap is
always one full circuit from wherever the car was placed.

**Evaluation.** Deterministic evaluation always launches from the canonical start
line with $v_0 = 0$:

$$ s_0 = (x_0, y_0, \theta_0, 0, 0, C) $$

so that every reported number answers the same question and is reproducible from
the seed alone. Because both the policy and the start are deterministic, one
evaluation episode per checkpoint is sufficient on a fixed circuit; repeating it
would produce identical episodes.

### Random Noise

If we later want to model a noisy environment, Gaussian noise can be added to the
four dynamic components:

$$ s_{t+1} = f(s_t, a_t) + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \Sigma) $$

## Reward Function

$$
r_t(s_t, s_{t+1}) =
\begin{cases}
-R_{\text{crash}} & \text{if } s_{t+1} \in \mathcal{W} \\
R_{\text{finish}} + R_{\text{lap}}\left(1 - \dfrac{t}{T_{\max}}\right) & \text{if } s_{t+1} \in \mathcal{F} \\
-c_{\text{step}}\left(T_{\max} - t + 1\right) & \text{if } s_{t+1} \in \mathcal{S} \\
-c_{\text{step}} + c_{\text{prog}} \cdot \Delta\tilde{s}_t & \text{otherwise}
\end{cases}
$$

Collision is checked before finish so crossing the gate while off-track cannot
receive the finish reward.

Where:
* $R_{\text{finish}} = 100$
* $R_{\text{lap}} = 100$ is the completion reward that lap time scales.
* $R_{\text{crash}} = 5$
* $c_{\text{step}}= \rho \cdot \Delta_{t_{agent}} = 0.04$, with $\rho = 1s^{-1}$ representing the cost over one agent step.
* $c_{\text{prog}}=100$
* $\Delta \tilde{s}_t$ is the progress term, computed as a normalized difference between the current and next locations (see [`TRACK.md`](TRACK.md)).

### Why these magnitudes

The coefficients are not free. They must preserve three orderings, or the task
becomes unlearnable, or learnable but not a race, by any policy-gradient method
regardless of its update rule.

**Trying must beat doing nothing.** A policy whose mean throttle is non-positive
never moves: at $v=0$ the pose and heading cannot change, so the observation and
reward are constant. That stalled policy earns $-c_{\text{step}}T_{\max}$. Any
exploratory attempt to drive risks $-R_{\text{crash}}$. If
$R_{\text{crash}} > c_{\text{step}}T_{\max}$, the stall strictly dominates every
attempt, and the stall is exactly the behaviour that gradient ascent can reach
from a neutral initialization. The requirement is therefore

$$ R_{\text{crash}} < c_{\text{step}} \cdot T_{\max}, $$

here $5 < 40$.

**Reaching further must be measurably better.** The shaped term is the only
signal that distinguishes a policy which crashes at 10% of the lap from one that
crashes at 90%. Its total authority over an episode is $c_{\text{prog}}$, against
a one-off $R_{\text{crash}}$. When $c_{\text{prog}} \ll R_{\text{crash}}$ the
return is, to within estimator noise, a function of *whether* the episode ended
in a crash and not of *where*, so there is no gradient toward driving further.
The requirement is

$$ c_{\text{prog}} \gg R_{\text{crash}}, $$

here $100 \gg 5$.

**A fast lap must beat a slow lap by a visible margin.** Progress and the step
penalty alone leave a crawling lap worth almost as much as an attacking one: a
lap three times slower loses only a twentieth of its return, so the task rewards
finishing rather than racing.

The obvious remedy, raising $\rho$ until the step penalty carries the
difference, cannot work. Raising it also raises what a *failed* attempt pays
before crashing, and past roughly $\rho = 2s^{-1}$ a policy that drives half a
lap and then crashes scores below one that crashes immediately, re-inverting the
first ordering. A term that only ever applies on success has no such side
effect, which is why $R_{\text{lap}}$ scales with the unused episode clock rather
than the time penalty being increased further. The requirement is

$$ \frac{r(\text{fast lap}) - r(\text{slow lap})}{r(\text{fast lap})} > 0.2, $$

evaluated over the range of lap times policies actually drive.

`experiments/plot_reward.py` draws all three orderings, next to the coefficients
this project started from, for which every one of them was violated.

An earlier version of this document used $R_{\text{finish}}=10$,
$R_{\text{crash}}=20$, $\rho=0.05s^{-1}$ and $c_{\text{prog}}=1$, with no lap
term. Under those values a policy had to already complete roughly two thirds of
its laps before attempting the lap beat standing still in expectation, which no
algorithm can bootstrap from a $0\%$ completion rate. REINFORCE, A2C and PPO all
converged to the stall.

The maximum episode length is $T_{\max}=1000$ agent steps, corresponding to
$40s$. With the friction budget in force, the deterministic reference controller
laps a generated circuit in about $16s$, or $400$ agent steps, so the cap keeps a
safety margin of roughly two and a half times the target lap. Both the circuit
scale and the cap are configuration values and can be increased together for a
longer task.

Here is an example of what may be going on with these values:

* The step penalty over a $16s$ lap is
  $-c_{\text{step}}\times 16/\Delta_{t_{agent}}=-16$.
* The normalized progress accumulated over one forward lap is approximately
  $+100$.
* The completion reward for a $16s$ lap is $100 + 100\times(1-0.4)=+160$.
* A $16s$ lap therefore returns approximately $100+160-16=244$. The exact value
  differs by at most one shaped transition because the terminal branch replaces
  the normal step reward.
* A $28s$ lap returns approximately $100+130-28=202$, seventeen percent less for
  the same completed circuit.
* Never leaving the start line returns $-c_{\text{step}}\times T_{\max}=-40$, the
  worst outcome available, whether the episode is truncated at $T_{\max}$ or
  ended early by the stall rule.
* An immediate crash returns $-R_{\text{crash}}=-5$.
* Crashing after one quarter of the lap returns approximately
  $100\times0.25-5-c_{\text{step}}\times T_{\text{crash}}\approx+16$, which is
  above both the stall and the immediate crash. The return is monotone in how far
  the car got.

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
