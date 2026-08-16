# Environment

This file contains information on how the racing track is represented and how such representation should be used throughout the project.

## Track Representation

The circuit is represented internally as a closed, **arc-length-parametrized centerline $\gamma$**. 
Given a distance $s$ along the centerline, $\gamma(s)$ returns a point on the 2D plane:

$$
\gamma(s) = (x(s), y(s)), \quad s \in [0,S_{\text{track}}), \qquad
\gamma(s+S_{\text{track}})=\gamma(s).
$$

At each distance $s$ along the circuit, we also care about:

* The **associated heading** $\psi(s)$, which is the direction the track is pointing at the location given by $s$. It can be measured as the tangent vector to $\gamma(s)$ in $(x(s), y(s))$:

    $$ \psi(s) = \text{atan2}\left(\frac{dy}{ds}, \frac{dx}{ds}\right) $$

* The **local curvature** $\kappa(s)=d\psi/ds$, which is the way the track curves around the location given by $s$. 
  The definition of *locality* considers the current velocity of the car: a faster car uses a longer **lookahead** $l$ to compute $\kappa(s)$; a slower car uses a shorter lookahead $l$:
  $$
  \ell(v_t)=5+0.7v_t,
  \qquad
  \bar{\kappa}_t =
  \frac{1}{\ell(v_t)}
  \int_0^{\ell(v_t)}
  \kappa((s_t+u)\bmod S_{\text{track}})\,du.
  $$

  At $v_t\in[0,70]m/s$, the lookahead varies from $5m$ to $54m$. 
  Since $\bar{\kappa}(s)$ cannot be pre-computed, as it depends on $v$, we precompute $\kappa(s)$ and use that to compute $\bar{\kappa}(s)$ on the fly.
  
The track has a constant width of $w=12m$.
The left and right boundaries are:

$$ \text{boundary}_{\pm}(s) = \gamma(s) \pm \frac{w}{2}\hat{n}(s) $$

where $\hat{n}(s)$ is the unit vector pointing to the left of the tangent:

$$ \hat{n}(s) = \left(-\sin\psi(s), +\cos\psi(s)\right) $$

## Track Generation and Table Representation

A circuit is built as **a sequence of straights and constant-radius corners**:

1. **Sample a closed polygon.** 
  Place `n_corners` vertices around one centre, in the shape of a star; then, apply both angular and radial jitters to offset them.

2. **Compute the rounding radius for each vertex.**
  At a vertex whose direction changes by $\Delta$ degrees, a corner of radius $R$ meets each of the two edges at a **tangent distance** $T$ from the vertex: $$ T = R\tan\left(\frac{|\Delta|}{2}\right) $$
  Each corner may claim at most **half** of each edge it touches.

3. **Choose each radius as a fraction of what fits.** 
  The radius is drawn as $R=\rho R_{\max}$, where $R_{\max}$ is the largest radius the two adjacent edges admit and $\rho$ is uniformly picked in the range $[\texttt{min\_corner\_radius},\texttt{max\_corner\_radius}]$. 

4. **Emit the segment list.** 
  Walking the polygon gives, for each vertex in turn, its arc followed by whatever survives of the outgoing edge.
  Doing this for each vertex results in walking the full circuit until back to the starting vertex.

5. **Scale to a whole number of samples.** 
  The total length is slighly rescaled so it is an exact multiple of $\Delta s_{\text{gen}}$. 

6. **Put the start/finish seam in the middle of the longest straight.** 
  The canonical start and finish line sits in the middle of the longest straight.
  This is used for deterministic evaluation, without random starts, and for the effective table representation of the track. 

#### Defaults

| Parameter | Default | Meaning |
|---|---:|---|
| `n_corners` | 9 | Corners, and so polygon vertices |
| `base_radius` | $70m$ | Radius the polygon is sampled around |
| `radial_jitter` | $\pm55\%$ | Independent vertex-radius variation |
| `angular_jitter` | $\pm\frac{3}{10}$ sector | Variation from equally spaced vertex angles |
| `corner_radius_fraction` | $[0.25,0.80]$ | Corner radius as a fraction of the largest that fits |
| `min_corner_radius` | $12m$ | Tightest corner the generator may produce |
| `max_corner_radius` | $200m$ | Most open corner the generator may produce |
| $\Delta s_{\text{gen}}$ | $0.5m$ | Spacing of the final lookup table |
| $w$ | $12m$ | Constant version 0 track width |
| `max_attempts` | 100 | Deterministic retries before generation fails |

Every generation request requires an integer seed. Retries derive their random
streams deterministically from that seed, so the same configuration and seed
produce byte-equivalent track data with the same generator version.

#### Conditions for Successful Generation

Generated tracks must satisfy all of the following:

1. The centerline is a simple closed curve: non-adjacent centerline segments do not intersect.
2. The seam is periodic in position, tangent and curvature. The final serialized table does not duplicate the first point; the closing segment is implicit.
3. Every sampled curvature respects the vehicle's kinematic minimum turning radius:
  $$ \frac{1}{|\kappa(s)|} \ge R_{\min} = \frac{L}{\tan(\delta_{max})} \quad \forall s $$
4. The distance between non-neighbouring centerline segments is greater than $w+2m$, preventing overlapping boundaries and leaving a $1m$ margin on both sides. The rule therefore ignores pairs of samples closer together along the track than $\pi\,\texttt{min\_corner\_radius}$, so that tight corners are still possible.
5. The left and right boundaries are themselves simple closed curves and never intersect each other.
6. The total length is between $300m$ and $700m$. This configurable training scale keeps a lap well inside the $T_{\max}$ episode cap.

If a generated candidate fails any validation, the generator retries until `max_attempts` and then raises an error.

#### Lookup Table Representation

The final dense lookup table stores:

$$ \texttt{table}[i] = (s_i, x_i, y_i, \psi_i, \kappa_i) $$

Every column is evaluated in closed form from the segment the sample falls in:

* $s_i=i\Delta s_{\text{gen}}$.
* $(x_i, y_i)$ and $\psi_i$ come from advancing the segment's start pose by the sample's offset into it. 
* $\kappa_i$ is the curvature of that segment: exactly $0$ on a straight and
  exactly $\pm1/R$ in a corner.

## Using the Track

### Car Pose to Frenet Pose

At each simulation step, the car's Cartesian pose $p = (x_t, y_t)$ must be translated into the Frenet pose $(s_t, d_t)$, with:
* $s_t$ being the arc-length distance along the centerline between the start of the circuit and the closest centerline point to $(x_t, y_t)$.
* $d_t$ being the signed lateral offset.

We therefore need a way to compute the mapping:
$$ (x_t, y_t) \longrightarrow (s_t, d_t) $$

The projection is onto the closest point of a **centerline segment**. 
For a candidate segment from $p_i$ to $p_{i+1}$, compute the clamped projection parameter

$$
u =
\operatorname{clip}_{[0,1]}
\frac{(p-p_i)^\top(p_{i+1}-p_i)}{\lVert p_{i+1}-p_i\rVert^2},
\qquad
p_{\text{proj}}=p_i+u(p_{i+1}-p_i).
$$

The associated $s$ is where that segment starts, plus the same fraction $u$ of one sample interval, wrapped to $[0,S_{\text{track}})$:
$$ s = \left((i+u)\,\Delta s_{\text{gen}}\right) \bmod S_{\text{track}}. $$

The signed lateral offset is:
$$ d=(p-p_{\text{proj}})^\top\hat n(s). $$

To keep this cheap, we use a temporally coherent segment window around the previous projection. 
Its half-width in samples is $\lceil v_{\max}\Delta t_{\mathrm{phys}}/\Delta s_{\mathrm{gen}}\rceil+4$.
Resets, teleports and failed local searches use a global spatial index over all centerline segments. 

### Computing the Frenet Observation

$$ o_t^{\text{Frenet}} = (d_t, \phi_{e,t}, v_t, \delta_t, \bar{\kappa}_t) $$

computed as:

$$
\phi_{e,t} =
\operatorname{wrap}_{[-\pi,\pi)}(\theta_t-\psi(s_t)),
\qquad
\bar{\kappa}_t = \bar{\kappa}(s_t,v_t).
$$

Where the missing measures $v_t, \delta_t, \theta_t$ are parts of the car's state, stored over time.

### Computing the LiDAR Observation

$$
o_t^{\text{LiDAR}} = (v_t, \delta_t, R_t), \qquad
R_t = (\tilde r_t^{(1)}, \dots, \tilde r_t^{(16)}).
$$

The ray offsets relative to car heading are inclusive and evenly spaced:

$$
\alpha_k=-100°+k\frac{200°}{15}, \qquad k=0,\ldots,15.
$$

Ray $0$ looks $100°$ to the right and ray $15$ looks $100°$ to the left. 
Nothing behind the car within $80°$ of straight back is seen at all.

Each ray starts at the point-car pose, has a maximum range $r_{\max}=100m$, and returns the distance to its first boundary intersection.
A ray with no hit returns $r_{\max}$. 
The policy receives normalized ranges $\tilde r=r/r_{\max}\in[0,1]$; raw metre values may be included in the `info` dictionary for debugging.

Ray casting tests **all** left and right boundary segments, in one vectorized pass over every (ray, segment) pair. 
It must not be restricted by arc-length proximity, because a ray can see a geometrically nearby section that is far away along the lap.
The only segments discarded are those whose distance from the vehicle exceeds $r_{\max}$ plus half a segment, which no ray could reach whatever its direction. 

### Collision Detection

Given the projection $(s_t, d_t)$, the car is off-track if:

$$ |d_t| \ge \frac{w}{2}. $$

This rule treats the car as a point and is evaluated after every simulation step.

### Progress Term of the Reward

Let $s_t^{\text{wrap}}\in[0,S_{\text{track}})$ be the projected position. 
The signed change is:

$$
\Delta s_t =
\operatorname{wrap}_{[-S_{\text{track}}/2,S_{\text{track}}/2)}
\left(s_t^{\text{wrap}}-s_{t-1}^{\text{wrap}}\right).
$$

This is positive when moving forward and negative when moving backward.

$$
s_t^{\text{episode}}=s_{t-1}^{\text{episode}}+\Delta s_t,
\qquad
s_0^{\text{episode}}=0.
$$

Because the vehicle cannot travel half a circuit in one step, the periodic wrap has an unambiguous branch.
The normalized reward term is:

$$ \Delta\tilde{s}_t = \frac{\Delta s_t}{S_{\text{track}}}, $$

with $\Delta\tilde{s}_0=0$ on reset.

### Finish Line and Lap Completion

The finish gate of an episode is the segment joining the two boundaries at the arc length the car started from, oriented by the forward tangent there. 
A forward crossing requires the car trajectory to intersect this segment and have positive motion along that tangent.
Because the start pose is sampled (see [`MDP.md`](MDP.md)), the gate moves with it.

An episode finishes only when all of the following hold:

1. The car crosses the finish gate in the forward direction.
2. It is still on track after the crossing.
3. Its signed episode progress is at least
   $S_{\text{track}}-\epsilon_{\text{finish}}$, where
   $\epsilon_{\text{finish}}=\max(2\Delta s_{\text{gen}},
   v_{\max}\Delta_{t_{agent}})=2.8m$ with the current constants.

The progress requirement prevents the reset pose on the finish line from looking like a lap completion.
If collision and finish are detected in the same physics substep, collision takes precedence.
