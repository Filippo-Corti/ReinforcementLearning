# Environment

This file contains information on how the racing track is represented and how such representation should be used throughout the project.

## Track Representation

The circuit is represented internally as an **arc-length-parametrized centerline $\gamma$**. Given a certain distance $s$ traveled along the circuit's centerline, the function $\gamma(s)$ returns the $x$ and $y$ coordinates of that point on a 2D plane:

$$ \gamma(s) = (x(s), y(s)), \quad s \in [0, S_{\text{track}}] $$

At each distance $s$ along the circuit, we also care about:

* The associated heading $\psi(s)$, which is the direction the track is pointing at the location given by $s$. It can be measured as the tangent vector to $\gamma(s)$ in $(x(s), y(s))$:

    $$ \phi(s) = \text{atan2}\left(\frac{dy}{ds}, \frac{dx}{ds}\right) $$

* The curvature $\kappa(s)$, which represents the way the heading $\psi(s)$ of the track changes as we move forward from $s$. We can compute it as the difference in heading $\psi$ between $s$ and a point ahead of it. To choose how far ahead the point should be, we consider the current velocity of the car:

    $$ \kappa(s) \approx \frac{\phi(s + \ell_{\text{lookahead}}) - \phi(s)}{\ell_{\text{lookahead}}} $$

    $$ \ell_{\text{lookahead}} = 5 + 0.7 v_t $$

We assume the track to have equal width $w$ everywhere. This means that the left and right boundaries of the track, in a given location $s$ can be computed as:

$$ \text{boundary}_{\pm}(s) = \gamma(s) \pm \frac{w}{2}\hat{n}(s) $$

Where $\hat{n}(s)$ is the unit vector perpendicular to the tangent $\phi(s)$:
$$ \hat{n}(s) = \left(-\sin\psi(s), +\cos\psi(s)\right) $$

This single parametrization is the ground-truth object the environment holds; both
observation types (Frenet, LiDAR) and all derived quantities (*e.g.*, collision, progress) are
computed from it, rather than being independently implemented.

### Track Generation and Table Representation

Track geometry is generated using the **checkpoint-and-smoothing algorithm** adapted from OpenAI Gym / Gymnasium's `CarRacing-v2`: 
random checkpoints are scattered around a circle with angular and radial jitter, then a heading $\beta$ is walked around the loop under a bounded turn rate, advancing by a fixed arc-length step $\Delta s_{\text{gen}}$ at each iteration. 
An additional validity check, absent from the original algorithm, is applied at generation time: every sampled curvature must respect the vehicle's minimum turning radius

$$ \frac{1}{|\kappa(s)|} \ge R_{\min} = \frac{L}{\tan(\delta_{max})} \quad \forall s $$

This constraint is imposed during the generation, to ensure that only completable circuits are generated.

This procedure allows for a dense lookup table to be used as a practical representation of the circuit shape $\gamma(s)$.
In fact, as the generation builds the track step-by-step, with a step size $\Delta s_{\text{gen}}$, we can easily store:

$$ \texttt{table}[i] = (s_i, x_i, y_i, \psi_i, \kappa_i) $$

In particular:
* $s_i$ is simply $s_i = i \cdot \Delta s_{\text{gen}}$.
* $(x_i, y_i)$ are recorded directly during the walk.
* $\psi_i$ is also recorded during the walk, as it is just $\psi = \beta$.
* $\kappa_i$ can be computed given three consecutive entries in the table:
    $$ \kappa_i = \frac{\text{wrap}(\psi_{i+1} - \psi_{i-1})}{2 \Delta s_{\text{gen}}} $$
  with $\text{wrap}$ making sure that the angles wrap around nicely (*e.g.*, from $179°$ to $-179°$ the difference is $2°$ and not $358°$).

## Track Usage

At each step, the car's Cartesian pose $(x_t, y_t)$ must be translated into the Frenet pose $(s_t, d_t)$, with:
* $s_t$ being the arc-length distance along the centerline between the start of the circuit and the closest centerline point to $(x_t, y_t)$.
* $d_t$ being the lateral offset from $s_t$ (positive or negative).

We therefore need a way to compute the mapping:
$$ (x_t, y_t) \longrightarrow (s_t, d_t) $$

To keep this cheap, we use a **temporally coherent** search on the lookup table: since the car cannot teleport, the projection searches only a small window of the table around the previous step's index, rather than the full table.
The search stops when the closest $\texttt{table}[i]$ - with its $(x_i, y_i)$ location - is found. 

The only case where a global search is needed is if we choose to have episodes start at random locations.

### Frenet Observation

$$ o_t^{\text{Frenet}} = (d_t, \phi_{e,t}, v_t, \kappa_t) $$

computed as:

$$ \phi_{e,t} = \theta_t - \psi(s_t), \qquad \kappa_t = \kappa(s_t) $$

where $\theta_t$ is the car's heading (from the true environment state).

### LiDAR Observation

$$ o_t^{\text{LiDAR}} = (v_t, R_t), \qquad R_t = (r_t^{(1)}, \dots, r_t^{(n)}) $$

Each ray $r_t^{(k)}$ is cast from the car's pose at a fixed angular offset and intersected against the nearby boundary segments (again restricted to the local window around the car's current table index, for the same performance reason as the projection search), returning the distance to first contact with $\text{boundary}_{\pm}(s)$. 
LiDAR reflects only solid track boundaries: there is no separate off-road buffer layer.

### Collision Detection

Given the projection $(s_t, d_t)$, the car is off-track if:

$$ |d_t| > \frac{w}{2} $$


### Progress Term of the Reward

Progress is the (signed) delta in arc-length position between consecutive steps, normalized by total track length:

$$ \Delta\tilde{s}_t = \frac{s_t - s_{t-1}}{S_{\text{track}}} $$

$s_t$ is tracked as a monotonically increasing, unwrapped quantity within an episode (reset to $\Delta\tilde{s}_0 = 0$ at episode start) to avoid a spurious large negative delta from wrap-around at the finish line; lap completion is instead handled as the separate terminal condition defined above. 
